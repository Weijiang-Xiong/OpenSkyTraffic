"""CleanRL-style PPO training pipeline for SkyMonitor drone monitoring.

This reproduces the PPO workflow in `skymonitor/rl_drone.py` without SB3's training loop.
"""

import argparse
import json
import logging
import random
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from skytraffic.utils.event_logger import setup_logger
from skytraffic.utils.io import make_dir_if_not_exist
from skymonitor.agents import BaseAgent
from skymonitor.monitor_env import MapStructure, TrafficMonitorEnv, build_traffic_monitor_env, eval_on_all_sessions
from skymonitor.policy_net import GraphFeatureExtractor, GridCNNFeatureExtractor, SimpleFeatureExtractor
from skymonitor.simbarca_explore import initialize_dataset

POSITION_KEYS = {'positions_x', 'positions_y'}
SB3_ROLLOUT_STEPS = 2048
SB3_MINIBATCH_TARGET = 128


def parse_args():
	parser = argparse.ArgumentParser(description='Train and evaluate CleanRL-style PPO monitoring agent.')
	parser.add_argument('--eval-only', action='store_true', help='If set, skip training and only run evaluation.')
	parser.add_argument('--policy-type', type=str, default='simple', choices=['simple', 'graph', 'grid'])
	parser.add_argument('--train-timesteps', type=int, default=int(1e6), help='Total environment steps for training.')
	parser.add_argument('--num-envs', type=int, default=8, help='Number of parallel environments.')
	parser.add_argument(
		'--num-steps',
		type=int,
		default=0,
		help='Rollout length per environment. Set <=0 to use SB3 parity default max(32, 2048 // num_envs).',
	)
	parser.add_argument(
		'--num-minibatches',
		type=int,
		default=0,
		help='Number of minibatches per PPO epoch. Set <=0 to auto-resolve near minibatch size 128.',
	)
	parser.add_argument('--update-epochs', type=int, default=10, help='Number of PPO epochs per update.')
	parser.add_argument('--num-drones', type=int, default=10, help='Number of drones.')
	parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')
	parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor.')
	parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda.')
	parser.add_argument('--clip-coef', type=float, default=0.2, help='PPO clip coefficient.')
	parser.add_argument('--clip-vloss', action=argparse.BooleanOptionalAction, default=False, help='Enable clipped value loss.')
	parser.add_argument('--ent-coef', type=float, default=0.0, help='Entropy coefficient.')
	parser.add_argument('--vf-coef', type=float, default=0.5, help='Value loss coefficient.')
	parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Gradient clipping norm.')
	parser.add_argument('--target-kl', type=float, default=None, help='Early stop update if approx KL exceeds this.')
	parser.add_argument('--anneal-lr', action=argparse.BooleanOptionalAction, default=False, help='Use linear LR annealing.')
	parser.add_argument('--reward-norm', action=argparse.BooleanOptionalAction, default=True, help='Enable SB3-style reward normalization.')
	parser.add_argument('--clip-reward', type=float, default=10.0, help='Clip normalized rewards to [-clip_reward, clip_reward].')
	parser.add_argument('--norm-adv', action=argparse.BooleanOptionalAction, default=True, help='Normalize advantages.')
	parser.add_argument('--norm-obs', action=argparse.BooleanOptionalAction, default=True, help='Normalize observations in environment.')
	parser.add_argument('--eval-freq', type=int, default=int(1e4), help='Eval frequency, every eval_freq * num_envs environment steps.')
	parser.add_argument('--eval-repeat', type=int, default=5, help='Number of repeated eval runs.')
	parser.add_argument('--eval-seed', type=int, default=888, help='Seed used for eval repeats.')
	parser.add_argument('--eval-deterministic', action=argparse.BooleanOptionalAction, default=False, help='Use greedy actions in evaluation.')
	parser.add_argument('--logdir', type=str, default='scratch/drone_monitor_cleanrl', help='Output directory for logs and checkpoints.')
	parser.add_argument('--save-name', type=str, default='model_cleanrl', help='Checkpoint basename (without extension).')
	parser.add_argument('--log-level', type=str, default='info', choices=['debug', 'info', 'warning', 'error', 'critical'])
	parser.add_argument('--train-seed', type=int, default=42, help='Training seed.')
	parser.add_argument('--hidden-dim', type=int, default=256, help='Encoder hidden dimension.')
	parser.add_argument('--policy-hidden-dim', type=int, default=256, help='Policy MLP hidden dimension.')
	parser.add_argument('--value-hidden-dim', type=int, default=256, help='Value MLP hidden dimension.')
	parser.add_argument('--cuda', action=argparse.BooleanOptionalAction, default=True, help='Enable CUDA when available.')
	parser.add_argument('--torch-deterministic', action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument('--capture-video', action='store_true', help='Record videos for env 0 during training.')
	return parser.parse_args()


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
	torch.nn.init.orthogonal_(layer.weight, std)
	if layer.bias is not None:
		torch.nn.init.constant_(layer.bias, bias_const)
	return layer


def to_torch_obs(obs_np: Dict[str, np.ndarray], device: torch.device, add_batch_dim: bool = False) -> Dict[str, torch.Tensor]:
	obs_t = {}
	for key, value in obs_np.items():
		tensor = torch.as_tensor(value, device=device)
		if key in POSITION_KEYS:
			tensor = tensor.long()
		else:
			tensor = tensor.float()
		if add_batch_dim:
			tensor = tensor.unsqueeze(0)
		obs_t[key] = tensor
	return obs_t


def zeros_like_obs_buffer(
	observation_space: spaces.Dict,
	num_steps: int,
	num_envs: int,
	device: torch.device,
) -> Dict[str, torch.Tensor]:
	buffers: Dict[str, torch.Tensor] = {}
	for key, space in observation_space.spaces.items():
		dtype = torch.long if key in POSITION_KEYS else torch.float32
		buffers[key] = torch.zeros((num_steps, num_envs) + space.shape, dtype=dtype, device=device)
	return buffers


def build_encoder(
	observation_space: spaces.Dict,
	policy_type: str,
	map_structure: Optional[MapStructure],
	hidden_dim: int,
) -> nn.Module:
	if policy_type == 'simple':
		return SimpleFeatureExtractor(observation_space=observation_space, hidden_dim=hidden_dim)
	if policy_type == 'graph':
		if map_structure is None:
			raise ValueError('map_structure is required for graph policy.')
		return GraphFeatureExtractor(
			observation_space=observation_space,
			adjacency_matrix=map_structure.adjacency_matrix,
			hidden_dim=hidden_dim,
			gnn_hidden_dim=hidden_dim,
		)
	if policy_type == 'grid':
		if map_structure is None:
			raise ValueError('map_structure is required for grid policy.')
		return GridCNNFeatureExtractor(
			observation_space=observation_space,
			map_structure=map_structure,
			hidden_dim=hidden_dim,
		)
	raise ValueError(f'Unsupported policy_type: {policy_type}')


class RunningMeanStd:
	def __init__(self, epsilon: float = 1e-4):
		self.mean = np.float64(0.0)
		self.var = np.float64(1.0)
		self.count = float(epsilon)

	def update(self, values: np.ndarray) -> None:
		values = np.asarray(values, dtype=np.float64)
		if values.size == 0:
			return
		batch_mean = float(values.mean())
		batch_var = float(values.var())
		batch_count = int(values.shape[0])
		self._update_from_moments(batch_mean, batch_var, batch_count)

	def _update_from_moments(self, batch_mean: float, batch_var: float, batch_count: int) -> None:
		delta = batch_mean - self.mean
		total_count = self.count + batch_count
		new_mean = self.mean + delta * batch_count / total_count
		m_a = self.var * self.count
		m_b = batch_var * batch_count
		m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count
		self.mean = new_mean
		self.var = m2 / total_count
		self.count = total_count


class RewardNormalizer:
	def __init__(
		self,
		num_envs: int,
		gamma: float = 0.99,
		clip_reward: float = 10.0,
		epsilon: float = 1e-8,
	):
		self.returns = np.zeros(num_envs, dtype=np.float64)
		self.gamma = float(gamma)
		self.clip_reward = float(clip_reward)
		self.epsilon = float(epsilon)
		self.ret_rms = RunningMeanStd()

	def normalize(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
		rewards = np.asarray(rewards, dtype=np.float64)
		dones = np.asarray(dones, dtype=bool)
		self.returns = self.returns * self.gamma + rewards
		self.ret_rms.update(self.returns)
		rewards = rewards / np.sqrt(self.ret_rms.var + self.epsilon)
		rewards = np.clip(rewards, -self.clip_reward, self.clip_reward)
		self.returns[dones] = 0.0
		return rewards.astype(np.float32)


def resolve_num_steps(num_envs: int, num_steps: int) -> int:
	if num_steps > 0:
		return int(num_steps)
	return max(32, SB3_ROLLOUT_STEPS // max(1, num_envs))


def resolve_num_minibatches(batch_size: int, num_minibatches: int, target_minibatch_size: int = SB3_MINIBATCH_TARGET) -> int:
	if num_minibatches > 0:
		if batch_size % num_minibatches != 0:
			raise ValueError('batch_size must be divisible by num_minibatches.')
		return int(num_minibatches)
	divisors = [d for d in range(1, batch_size + 1) if batch_size % d == 0]
	if not divisors:
		raise ValueError(f'No valid divisor found for batch_size={batch_size}.')
	return min(divisors, key=lambda d: (abs((batch_size // d) - target_minibatch_size), -d))


class DronePPOAgent(nn.Module):
	@staticmethod
	def init_weights(module: nn.Module, gain: float = 1.0) -> None:
		if isinstance(module, (nn.Linear, nn.Conv2d)):
			nn.init.orthogonal_(module.weight, gain=gain)
			if module.bias is not None:
				nn.init.constant_(module.bias, 0.0)

	def __init__(
		self,
		observation_space: spaces.Dict,
		action_space: spaces.MultiDiscrete,
		policy_type: str,
		map_structure: Optional[MapStructure],
		hidden_dim: int = 256,
		policy_hidden_dim: int = 256,
		value_hidden_dim: int = 256,
	):
		super().__init__()
		if not isinstance(action_space, spaces.MultiDiscrete):
			raise ValueError('DronePPOAgent currently supports only MultiDiscrete action spaces.')

		self.action_dims = [int(x) for x in action_space.nvec]
		self.num_action_heads = len(self.action_dims)

		self.encoder = build_encoder(
			observation_space=observation_space,
			policy_type=policy_type,
			map_structure=map_structure,
			hidden_dim=hidden_dim,
		)
		self.encoder.apply(partial(self.init_weights, gain=np.sqrt(2)))
		feature_dim = int(self.encoder.features_dim)

		self.policy_net = nn.Sequential(
			layer_init(nn.Linear(feature_dim, policy_hidden_dim)),
			nn.ReLU(),
			layer_init(nn.Linear(policy_hidden_dim, policy_hidden_dim)),
			nn.ReLU(),
		)
		self.value_net = nn.Sequential(
			layer_init(nn.Linear(feature_dim, value_hidden_dim)),
			nn.ReLU(),
			layer_init(nn.Linear(value_hidden_dim, value_hidden_dim)),
			nn.ReLU(),
		)
		self.actor_heads = nn.ModuleList(
			[layer_init(nn.Linear(policy_hidden_dim, n_actions), std=0.01) for n_actions in self.action_dims]
		)
		self.critic = layer_init(nn.Linear(value_hidden_dim, 1), std=1.0)

	def get_value(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
		features = self.encoder(observations)
		return self.critic(self.value_net(features))

	def get_action_and_value(
		self,
		observations: Dict[str, torch.Tensor],
		action: Optional[torch.Tensor] = None,
		deterministic: bool = False,
	):
		features = self.encoder(observations)
		policy_latent = self.policy_net(features)
		value_latent = self.value_net(features)

		dists = [Categorical(logits=head(policy_latent)) for head in self.actor_heads]
		if action is None:
			if deterministic:
				action = torch.stack([dist.probs.argmax(dim=-1) for dist in dists], dim=1)
			else:
				action = torch.stack([dist.sample() for dist in dists], dim=1)
		else:
			action = action.long()

		log_prob = torch.stack([dist.log_prob(action[:, idx]) for idx, dist in enumerate(dists)], dim=1).sum(dim=1)
		entropy = torch.stack([dist.entropy() for dist in dists], dim=1).sum(dim=1)
		value = self.critic(value_latent)

		return action, log_prob, entropy, value


class PolicyAgentCleanRL(BaseAgent):
	def __init__(self, policy: DronePPOAgent, device: torch.device, deterministic: bool = False):
		self.policy = policy
		self.device = device
		self.deterministic = deterministic

	def select_action(self, obs: Dict) -> np.ndarray:
		obs_t = to_torch_obs(obs, self.device, add_batch_dim=True)
		with torch.no_grad():
			actions, _, _, _ = self.policy.get_action_and_value(obs_t, deterministic=self.deterministic)
		return actions.squeeze(0).cpu().numpy()


def make_env(trainset, num_drones, norm_obs, seed, rank, capture_video, run_name):
	def init():
		env = build_traffic_monitor_env(
			trainset=trainset,
			testset=None,
			num_drones=num_drones,
			env_type='train',
			norm_obs=norm_obs,
		)
		env = gym.wrappers.RecordEpisodeStatistics(env)
		if capture_video and rank == 0:
			# TrafficMonitorEnv currently does not provide rgb-array rendering.
			pass
		env.reset(seed=seed + rank)
		return env

	return init


def periodic_evaluation(
	eval_env: TrafficMonitorEnv,
	policy: DronePPOAgent,
	device: torch.device,
	eval_repeat: int,
	eval_seed: int,
	deterministic: bool = False,
) -> Dict[str, float]:
	eval_agent = PolicyAgentCleanRL(policy=policy, device=device, deterministic=deterministic)
	eval_rewards = []

	rng = np.random.default_rng(eval_seed)
	seeds = rng.choice(10000, size=eval_repeat, replace=False)
	for seed in seeds:
		with torch.no_grad():
			eval_res = eval_on_all_sessions(eval_env, eval_agent, seed=int(seed))
		eval_rewards.append(float(np.mean(eval_res['all_reward'])))

	eval_rewards = np.asarray(eval_rewards, dtype=np.float32)
	return {
		'reward': float(eval_rewards.mean()),
		'std_reward': float(eval_rewards.std()),
	}


def final_evaluation(
	env: TrafficMonitorEnv,
	policy: DronePPOAgent,
	device: torch.device,
	eval_repeat: int,
	eval_seed: int,
	deterministic: bool = False,
):
	agent = PolicyAgentCleanRL(policy=policy, device=device, deterministic=deterministic)
	res_reps = defaultdict(list)

	rng = np.random.default_rng(eval_seed)
	seeds = rng.choice(10000, size=eval_repeat, replace=False)
	for seed in seeds:
		with torch.no_grad():
			eval_res = eval_on_all_sessions(env, agent, seed=int(seed))
		res_reps['reward'].append(eval_res['all_reward'])
		res_reps['trajectories'].append(eval_res['all_trajectories'])

	reward_reps = np.asarray(res_reps['reward'], dtype=np.float32)
	stats = {
		'avg_reward': float(reward_reps.mean(axis=1).mean()),
		'std_reward': float(reward_reps.mean(axis=1).std()),
		'avg_reward_per_session': reward_reps.mean(axis=0).tolist(),
		'std_reward_per_session': reward_reps.std(axis=0).tolist(),
	}
	return stats, reward_reps, res_reps['trajectories']


def train(args, logger):
	run_name = f"drone_cleanrl_{args.policy_type}__{args.train_seed}__{int(time.time())}"
	writer = SummaryWriter(log_dir=str(Path(args.logdir) / 'runs' / run_name))
	writer.add_text(
		'hyperparameters',
		'|param|value|\n|-|-|\n%s' % '\n'.join([f'|{k}|{v}|' for k, v in vars(args).items()]),
	)

	random.seed(args.train_seed)
	np.random.seed(args.train_seed)
	torch.manual_seed(args.train_seed)
	torch.backends.cudnn.deterministic = args.torch_deterministic

	device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

	args.num_steps = resolve_num_steps(num_envs=args.num_envs, num_steps=args.num_steps)
	args.batch_size = int(args.num_envs * args.num_steps)
	args.num_minibatches = resolve_num_minibatches(
		batch_size=args.batch_size,
		num_minibatches=args.num_minibatches,
		target_minibatch_size=SB3_MINIBATCH_TARGET,
	)
	args.minibatch_size = int(args.batch_size // args.num_minibatches)
	args.num_iterations = int(args.train_timesteps // args.batch_size)
	if args.num_iterations < 1:
		raise ValueError('train-timesteps is too small for current num-envs and num-steps.')
	logger.info(
		'Resolved PPO hyperparameters: '
		f'num_steps={args.num_steps}, '
		f'batch_size={args.batch_size}, '
		f'minibatch_size={args.minibatch_size}, '
		f'num_minibatches={args.num_minibatches}, '
		f'update_epochs={args.update_epochs}, '
		f'anneal_lr={args.anneal_lr}, '
		f'clip_vloss={args.clip_vloss}, '
		f'ent_coef={args.ent_coef}, '
		f'reward_norm={args.reward_norm}, '
		f'clip_reward={args.clip_reward}'
	)
	writer.add_text(
		'resolved_hyperparameters',
		'|param|value|\n|-|-|\n%s'
		% '\n'.join(
			[
				f'|num_steps|{args.num_steps}|',
				f'|batch_size|{args.batch_size}|',
				f'|minibatch_size|{args.minibatch_size}|',
				f'|num_minibatches|{args.num_minibatches}|',
				f'|update_epochs|{args.update_epochs}|',
				f'|anneal_lr|{args.anneal_lr}|',
				f'|clip_vloss|{args.clip_vloss}|',
				f'|ent_coef|{args.ent_coef}|',
				f'|reward_norm|{args.reward_norm}|',
				f'|clip_reward|{args.clip_reward}|',
			]
		),
	)

	trainset, testset = initialize_dataset()
	template_env: TrafficMonitorEnv = build_traffic_monitor_env(
		trainset=trainset,
		testset=None,
		num_drones=args.num_drones,
		env_type='train',
		norm_obs=args.norm_obs,
	)
	map_structure = template_env.map_structure
	obs_space = template_env.observation_space
	act_space = template_env.action_space
	template_env.close()

	envs = gym.vector.SyncVectorEnv(
		[
			make_env(
				trainset=trainset,
				num_drones=args.num_drones,
				norm_obs=args.norm_obs,
				seed=args.train_seed,
				rank=rank,
				capture_video=args.capture_video,
				run_name=run_name,
			)
			for rank in range(args.num_envs)
		]
	)

	assert isinstance(envs.single_observation_space, spaces.Dict), 'Only Dict observation is supported.'
	assert isinstance(envs.single_action_space, spaces.MultiDiscrete), 'Only MultiDiscrete action is supported.'

	agent = DronePPOAgent(
		observation_space=obs_space,
		action_space=act_space,
		policy_type=args.policy_type,
		map_structure=map_structure,
		hidden_dim=args.hidden_dim,
		policy_hidden_dim=args.policy_hidden_dim,
		value_hidden_dim=args.value_hidden_dim,
	).to(device)
	optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

	obs = zeros_like_obs_buffer(obs_space, args.num_steps, args.num_envs, device)
	actions = torch.zeros((args.num_steps, args.num_envs, len(act_space.nvec)), dtype=torch.long, device=device)
	logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
	rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
	raw_rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
	dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
	values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
	reward_normalizer = (
		RewardNormalizer(num_envs=args.num_envs, gamma=args.gamma, clip_reward=args.clip_reward)
		if args.reward_norm
		else None
	)

	global_step = 0
	start_time = time.time()
	next_obs_np, _ = envs.reset(seed=args.train_seed)
	next_obs = to_torch_obs(next_obs_np, device=device)
	next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)

	eval_env = build_traffic_monitor_env(
		trainset=trainset,
		testset=testset,
		num_drones=args.num_drones,
		env_type='test',
		norm_obs=args.norm_obs,
	)
	next_eval_step = args.eval_freq * args.num_envs if args.eval_freq > 0 else None

	logger.info(f'Starting training for {args.num_iterations} iterations.')

	for iteration in range(1, args.num_iterations + 1):
		if args.anneal_lr:
			frac = 1.0 - (iteration - 1.0) / args.num_iterations
			lr_now = frac * args.learning_rate
			optimizer.param_groups[0]['lr'] = lr_now

		for step in range(args.num_steps):
			global_step += args.num_envs
			for key in obs.keys():
				obs[key][step] = next_obs[key]
			dones[step] = next_done

			with torch.no_grad():
				action, logprob, _, value = agent.get_action_and_value(next_obs)
				values[step] = value.flatten()
			actions[step] = action
			logprobs[step] = logprob

			next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
			next_done_np = np.logical_or(terminations, truncations)
			raw_reward_np = np.asarray(reward, dtype=np.float32)
			raw_rewards[step] = torch.as_tensor(raw_reward_np, device=device, dtype=torch.float32)
			if reward_normalizer is not None:
				train_reward_np = reward_normalizer.normalize(raw_reward_np, next_done_np)
			else:
				train_reward_np = raw_reward_np
			rewards[step] = torch.as_tensor(train_reward_np, device=device, dtype=torch.float32)
			next_done = torch.as_tensor(next_done_np, device=device, dtype=torch.float32)
			next_obs = to_torch_obs(next_obs_np, device=device)

			if 'final_info' in infos:
				for info in infos['final_info']:
					if info and 'episode' in info:
						writer.add_scalar('charts/episodic_return', float(info['episode']['r']), global_step)
						writer.add_scalar('charts/episodic_length', float(info['episode']['l']), global_step)

		if next_eval_step is not None and global_step >= next_eval_step:
			eval_stats = periodic_evaluation(
				eval_env=eval_env,
				policy=agent,
				device=device,
				eval_repeat=args.eval_repeat,
				eval_seed=args.eval_seed,
				deterministic=args.eval_deterministic,
			)
			writer.add_scalar('eval/reward', eval_stats['reward'], global_step)
			writer.add_scalar('eval/std_reward', eval_stats['std_reward'], global_step)
			logger.info(
				f"step={global_step} eval_reward={eval_stats['reward']:.3f} eval_std={eval_stats['std_reward']:.3f}"
			)
			next_eval_step += args.eval_freq * args.num_envs

		with torch.no_grad():
			next_value = agent.get_value(next_obs).flatten()
			advantages = torch.zeros_like(rewards, device=device)
			lastgaelam = 0.0
			for t in reversed(range(args.num_steps)):
				if t == args.num_steps - 1:
					nextnonterminal = 1.0 - next_done
					nextvalues = next_value
				else:
					nextnonterminal = 1.0 - dones[t + 1]
					nextvalues = values[t + 1]
				delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
				advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
			returns = advantages + values

		b_obs = {key: value.reshape((-1,) + value.shape[2:]) for key, value in obs.items()}
		b_logprobs = logprobs.reshape(-1)
		b_actions = actions.reshape(-1, len(act_space.nvec))
		b_advantages = advantages.reshape(-1)
		b_returns = returns.reshape(-1)
		b_values = values.reshape(-1)

		b_inds = np.arange(args.batch_size)
		clipfracs = []
		for _ in range(args.update_epochs):
			np.random.shuffle(b_inds)
			for start in range(0, args.batch_size, args.minibatch_size):
				end = start + args.minibatch_size
				mb_inds = b_inds[start:end]
				mb_obs = {key: value[mb_inds] for key, value in b_obs.items()}

				_, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, b_actions[mb_inds])
				logratio = newlogprob - b_logprobs[mb_inds]
				ratio = logratio.exp()

				with torch.no_grad():
					old_approx_kl = (-logratio).mean()
					approx_kl = ((ratio - 1.0) - logratio).mean()
					clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

				mb_advantages = b_advantages[mb_inds]
				if args.norm_adv:
					mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

				pg_loss1 = -mb_advantages * ratio
				pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
				pg_loss = torch.max(pg_loss1, pg_loss2).mean()

				newvalue = newvalue.view(-1)
				if args.clip_vloss:
					v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
					v_clipped = b_values[mb_inds] + torch.clamp(
						newvalue - b_values[mb_inds],
						-args.clip_coef,
						args.clip_coef,
					)
					v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
					v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()
				else:
					v_loss = ((newvalue - b_returns[mb_inds]) ** 2).mean()

				entropy_loss = entropy.mean()
				loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

				optimizer.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
				optimizer.step()

			if args.target_kl is not None and approx_kl > args.target_kl:
				break

		y_pred, y_true = b_values.detach().cpu().numpy(), b_returns.detach().cpu().numpy()
		var_y = np.var(y_true)
		explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

		writer.add_scalar('charts/learning_rate', optimizer.param_groups[0]['lr'], global_step)
		writer.add_scalar('losses/value_loss', float(v_loss.item()), global_step)
		writer.add_scalar('losses/policy_loss', float(pg_loss.item()), global_step)
		writer.add_scalar('losses/entropy', float(entropy_loss.item()), global_step)
		writer.add_scalar('losses/old_approx_kl', float(old_approx_kl.item()), global_step)
		writer.add_scalar('losses/approx_kl', float(approx_kl.item()), global_step)
		writer.add_scalar('losses/clipfrac', float(np.mean(clipfracs) if clipfracs else 0.0), global_step)
		writer.add_scalar('losses/explained_variance', float(explained_var), global_step)
		writer.add_scalar('charts/rollout_reward_raw_mean', float(raw_rewards.mean().item()), global_step)
		writer.add_scalar('charts/rollout_reward_raw_std', float(raw_rewards.std().item()), global_step)
		writer.add_scalar('charts/rollout_reward_train_mean', float(rewards.mean().item()), global_step)
		writer.add_scalar('charts/rollout_reward_train_std', float(rewards.std().item()), global_step)
		if reward_normalizer is not None:
			writer.add_scalar(
				'charts/reward_norm_rms_std',
				float(np.sqrt(reward_normalizer.ret_rms.var + reward_normalizer.epsilon)),
				global_step,
			)
		writer.add_scalar('charts/SPS', int(global_step / (time.time() - start_time)), global_step)

	checkpoint_path = Path(args.logdir) / f'{args.save_name}.pt'
	torch.save(
		{
			'model_state_dict': agent.state_dict(),
			'policy_type': args.policy_type,
			'num_drones': args.num_drones,
			'hidden_dim': args.hidden_dim,
			'policy_hidden_dim': args.policy_hidden_dim,
			'value_hidden_dim': args.value_hidden_dim,
			'norm_obs': args.norm_obs,
		},
		checkpoint_path,
	)
	logger.info(f'Saved checkpoint: {checkpoint_path}')

	envs.close()
	eval_env.close()
	writer.close()


def load_policy_for_eval(args, device: torch.device, logger: logging.Logger):
	trainset, testset = initialize_dataset()
	template_env: TrafficMonitorEnv = build_traffic_monitor_env(
		trainset=trainset,
		testset=None,
		num_drones=args.num_drones,
		env_type='train',
		norm_obs=args.norm_obs,
	)
	map_structure = template_env.map_structure
	obs_space = template_env.observation_space
	act_space = template_env.action_space
	template_env.close()

	agent = DronePPOAgent(
		observation_space=obs_space,
		action_space=act_space,
		policy_type=args.policy_type,
		map_structure=map_structure,
		hidden_dim=args.hidden_dim,
		policy_hidden_dim=args.policy_hidden_dim,
		value_hidden_dim=args.value_hidden_dim,
	).to(device)

	ckpt_path = Path(args.logdir) / f'{args.save_name}.pt'
	ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
	agent.load_state_dict(ckpt['model_state_dict'])
	agent.eval()
	logger.info(f'Loaded checkpoint from {ckpt_path}')

	eval_env = build_traffic_monitor_env(
		trainset=trainset,
		testset=testset,
		num_drones=args.num_drones,
		env_type='test',
		norm_obs=args.norm_obs,
	)
	return agent, eval_env


def main():
	args = parse_args()
	make_dir_if_not_exist(args.logdir)
	logger = setup_logger(
		name='skymonitor.cleanrl',
		log_file=f"{args.logdir}/experiment_cleanrl.log",
		level=getattr(logging, args.log_level.upper()),
	)
	logger.info(f'Arguments: {args}')

	device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

	if not args.eval_only:
		train(args, logger)

	policy, eval_env = load_policy_for_eval(args, device=device, logger=logger)
	stats, reward_reps, trajectories = final_evaluation(
		env=eval_env,
		policy=policy,
		device=device,
		eval_repeat=args.eval_repeat,
		eval_seed=args.eval_seed,
		deterministic=args.eval_deterministic,
	)
	logger.info(f'Drone Monitoring Evaluation Results: {stats}')

	save_path = Path(args.logdir) / f'rep{args.eval_repeat}x_results.json'
	with open(save_path, 'w') as f:
		json.dump(
			{
				'stats': stats,
				'reward_all_repeats': reward_reps.tolist(),
				'trajectories_all_repeats': trajectories,
			},
			f,
			indent=4,
		)
	logger.info(f'Saved evaluation results to {save_path}')

	eval_env.close()


if __name__ == '__main__':
	main()
