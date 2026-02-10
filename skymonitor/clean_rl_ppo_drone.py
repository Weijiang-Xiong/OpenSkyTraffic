"""CleanRL-style PPO training pipeline for SkyMonitor drone monitoring.

This reproduces the PPO workflow in `skymonitor/rl_drone.py` without SB3's training loop.
"""

import argparse
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from gymnasium.spaces.utils import flatdim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from skytraffic.utils.event_logger import setup_logger
from skytraffic.utils.io import make_dir_if_not_exist
from skymonitor.agents import BaseAgent
from skymonitor.monitor_env import MapStructure, TrafficMonitorEnv, build_traffic_monitor_env, eval_on_all_sessions
from skymonitor.simbarca_explore import initialize_dataset

POSITION_KEYS = {'positions_x', 'positions_y'}


def parse_args():
	parser = argparse.ArgumentParser(description='Train and evaluate CleanRL-style PPO monitoring agent.')
	parser.add_argument('--eval-only', action='store_true', help='If set, skip training and only run evaluation.')
	parser.add_argument('--policy-type', type=str, default='simple', choices=['simple', 'graph', 'grid'])
	parser.add_argument('--train-timesteps', type=int, default=int(1e6), help='Total environment steps for training.')
	parser.add_argument('--num-envs', type=int, default=8, help='Number of parallel environments.')
	parser.add_argument('--num-steps', type=int, default=128, help='Rollout length per environment.')
	parser.add_argument('--num-minibatches', type=int, default=4, help='Number of minibatches per PPO epoch.')
	parser.add_argument('--update-epochs', type=int, default=4, help='Number of PPO epochs per update.')
	parser.add_argument('--num-drones', type=int, default=10, help='Number of drones.')
	parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')
	parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor.')
	parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda.')
	parser.add_argument('--clip-coef', type=float, default=0.2, help='PPO clip coefficient.')
	parser.add_argument('--clip-vloss', action=argparse.BooleanOptionalAction, default=True, help='Enable clipped value loss.')
	parser.add_argument('--ent-coef', type=float, default=0.01, help='Entropy coefficient.')
	parser.add_argument('--vf-coef', type=float, default=0.5, help='Value loss coefficient.')
	parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Gradient clipping norm.')
	parser.add_argument('--target-kl', type=float, default=None, help='Early stop update if approx KL exceeds this.')
	parser.add_argument('--anneal-lr', action=argparse.BooleanOptionalAction, default=True, help='Use linear LR annealing.')
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


def flatten_obs_batch(obs_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
	chunks = []
	for key in sorted(obs_batch.keys()):
		value = obs_batch[key]
		if key in POSITION_KEYS:
			value = value.float()
		chunks.append(value.reshape(value.shape[0], -1))
	return torch.cat(chunks, dim=1)


class SimpleDroneEncoder(nn.Module):
	def __init__(self, observation_space: spaces.Dict, hidden_dim: int = 256):
		super().__init__()
		base_dim = max(16, hidden_dim // 8)
		time_dim = max(8, hidden_dim // 32)
		pos_emb_dim = max(8, hidden_dim // 32)

		flow_in = int(flatdim(observation_space['flow']))
		density_in = int(flatdim(observation_space['density']))
		time_in = int(flatdim(observation_space['time_in_day']))
		coverage_in = int(flatdim(observation_space['coverage_mask']))

		num_x = int(observation_space['positions_x'].nvec.max())
		num_y = int(observation_space['positions_y'].nvec.max())

		self.flow_encoder = layer_init(nn.Linear(flow_in, base_dim))
		self.density_encoder = layer_init(nn.Linear(density_in, base_dim))
		self.time_encoder = layer_init(nn.Linear(time_in, time_dim))
		self.coverage_encoder = layer_init(nn.Linear(coverage_in, base_dim))
		self.x_embed = nn.Embedding(num_x, pos_emb_dim)
		self.y_embed = nn.Embedding(num_y, pos_emb_dim)

		self.ha_flow_encoder = None
		self.ha_density_encoder = None
		self.ha_flow_std_encoder = None
		self.ha_density_std_encoder = None
		extra_dim = 0
		if 'ha_flow' in observation_space.spaces:
			ha_flow_in = int(flatdim(observation_space['ha_flow']))
			self.ha_flow_encoder = layer_init(nn.Linear(ha_flow_in, base_dim))
			extra_dim += base_dim
		if 'ha_density' in observation_space.spaces:
			ha_density_in = int(flatdim(observation_space['ha_density']))
			self.ha_density_encoder = layer_init(nn.Linear(ha_density_in, base_dim))
			extra_dim += base_dim
		if 'ha_flow_std' in observation_space.spaces:
			ha_flow_std_in = int(flatdim(observation_space['ha_flow_std']))
			self.ha_flow_std_encoder = layer_init(nn.Linear(ha_flow_std_in, base_dim))
			extra_dim += base_dim
		if 'ha_density_std' in observation_space.spaces:
			ha_density_std_in = int(flatdim(observation_space['ha_density_std']))
			self.ha_density_std_encoder = layer_init(nn.Linear(ha_density_std_in, base_dim))
			extra_dim += base_dim

		concat_dim = (base_dim * 3) + time_dim + (2 * pos_emb_dim) + extra_dim
		self.proj = layer_init(nn.Linear(concat_dim, hidden_dim))
		self.out_dim = hidden_dim

	def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
		flow = observations['flow'].float().flatten(start_dim=1)
		density = observations['density'].float().flatten(start_dim=1)
		time_in_day = observations['time_in_day'].float().flatten(start_dim=1)
		coverage = observations['coverage_mask'].float().flatten(start_dim=1)

		flow_feat = torch.relu(self.flow_encoder(flow))
		density_feat = torch.relu(self.density_encoder(density))
		time_feat = torch.relu(self.time_encoder(time_in_day))
		coverage_feat = torch.relu(self.coverage_encoder(coverage))

		pos_x = observations['positions_x'].long()
		pos_y = observations['positions_y'].long()
		pos_x_emb = self.x_embed(pos_x).mean(dim=1)
		pos_y_emb = self.y_embed(pos_y).mean(dim=1)

		features = [flow_feat, density_feat, time_feat, coverage_feat, pos_x_emb, pos_y_emb]
		if self.ha_flow_encoder is not None:
			features.append(torch.relu(self.ha_flow_encoder(observations['ha_flow'].float().flatten(start_dim=1))))
		if self.ha_density_encoder is not None:
			features.append(torch.relu(self.ha_density_encoder(observations['ha_density'].float().flatten(start_dim=1))))
		if self.ha_flow_std_encoder is not None:
			features.append(torch.relu(self.ha_flow_std_encoder(observations['ha_flow_std'].float().flatten(start_dim=1))))
		if self.ha_density_std_encoder is not None:
			features.append(torch.relu(self.ha_density_std_encoder(observations['ha_density_std'].float().flatten(start_dim=1))))

		return torch.relu(self.proj(torch.cat(features, dim=1)))


class GraphDroneEncoder(nn.Module):
	def __init__(self, observation_space: spaces.Dict, map_structure: MapStructure, hidden_dim: int = 256):
		super().__init__()
		if map_structure is None:
			raise ValueError('map_structure is required for graph policy.')

		pos_emb_dim = max(8, hidden_dim // 32)
		time_dim = max(8, hidden_dim // 32)
		gnn_hidden_dim = hidden_dim

		adj = torch.as_tensor(map_structure.adjacency_matrix, dtype=torch.float32)
		adj = (adj > 0).float()
		adj = adj + torch.eye(adj.shape[0], dtype=adj.dtype)
		deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
		adj = adj / deg
		self.register_buffer('adjacency', adj)

		num_x = int(observation_space['positions_x'].nvec.max())
		num_y = int(observation_space['positions_y'].nvec.max())
		time_in = int(flatdim(observation_space['time_in_day']))

		self.node_encoder_1 = layer_init(nn.Linear(3, gnn_hidden_dim))
		self.node_encoder_2 = layer_init(nn.Linear(gnn_hidden_dim, gnn_hidden_dim))
		self.time_encoder = layer_init(nn.Linear(time_in, time_dim))
		self.x_embed = nn.Embedding(num_x, pos_emb_dim)
		self.y_embed = nn.Embedding(num_y, pos_emb_dim)
		self.proj = layer_init(nn.Linear(gnn_hidden_dim + time_dim + 2 * pos_emb_dim, hidden_dim))
		self.out_dim = hidden_dim

	def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
		flow = observations['flow'].float()
		density = observations['density'].float()
		coverage = observations['coverage_mask'].float()
		node_features = torch.stack([flow, density, coverage], dim=-1)

		node_hidden = torch.relu(self.node_encoder_1(node_features))
		node_hidden = torch.einsum('ij,bjk->bik', self.adjacency, node_hidden)
		node_hidden = torch.relu(self.node_encoder_2(node_hidden))
		graph_feat = node_hidden.mean(dim=1)

		time_feat = torch.relu(self.time_encoder(observations['time_in_day'].float().flatten(start_dim=1)))
		pos_x = observations['positions_x'].long()
		pos_y = observations['positions_y'].long()
		pos_x_emb = self.x_embed(pos_x).mean(dim=1)
		pos_y_emb = self.y_embed(pos_y).mean(dim=1)

		aux = torch.cat([time_feat, pos_x_emb, pos_y_emb], dim=1)
		return torch.relu(self.proj(torch.cat([graph_feat, aux], dim=1)))


class GridDroneEncoder(nn.Module):
	def __init__(self, observation_space: spaces.Dict, map_structure: MapStructure, hidden_dim: int = 256):
		super().__init__()
		if map_structure is None:
			raise ValueError('map_structure is required for grid policy.')

		grid_xy = torch.as_tensor(map_structure.grid_xy_of_nodes, dtype=torch.long)
		grid_width = int(grid_xy[:, 0].max().item()) + 1
		grid_height = int(grid_xy[:, 1].max().item()) + 1
		num_cells = grid_width * grid_height

		grid_index = grid_xy[:, 0] + (grid_xy[:, 1] * grid_width)
		grid_counts = torch.zeros(num_cells, dtype=torch.float32)
		grid_counts.scatter_add_(0, grid_index, torch.ones_like(grid_index, dtype=torch.float32))
		grid_counts = torch.clamp(grid_counts, min=1.0)

		grid_pos = torch.arange(num_cells, dtype=torch.long)
		grid_x = grid_pos % grid_width
		grid_y = grid_pos // grid_width

		pos_emb_dim = max(8, hidden_dim // 16)
		cnn_hidden_dim = hidden_dim
		time_dim = max(8, hidden_dim // 32)
		time_in = int(flatdim(observation_space['time_in_day']))

		self.grid_width = grid_width
		self.grid_height = grid_height
		self.num_cells = num_cells

		self.register_buffer('grid_index', grid_index)
		self.register_buffer('grid_counts', grid_counts)
		self.register_buffer('grid_x', grid_x)
		self.register_buffer('grid_y', grid_y)

		self.x_embed = nn.Embedding(self.grid_width, pos_emb_dim)
		self.y_embed = nn.Embedding(self.grid_height, pos_emb_dim)

		in_channels = 2 + (2 * pos_emb_dim)
		self.cnn = nn.Sequential(
			layer_init(nn.Conv2d(in_channels, cnn_hidden_dim, kernel_size=3, padding=1)),
			nn.ReLU(),
			layer_init(nn.Conv2d(cnn_hidden_dim, hidden_dim, kernel_size=3, padding=1)),
			nn.ReLU(),
		)
		self.time_encoder = layer_init(nn.Linear(time_in, time_dim))
		self.proj = layer_init(nn.Linear(hidden_dim + time_dim, hidden_dim))
		self.out_dim = hidden_dim

	def _grid_mean(self, values: torch.Tensor) -> torch.Tensor:
		batch_size = values.shape[0]
		idx = self.grid_index.unsqueeze(0).expand(batch_size, -1)
		grid = torch.zeros(batch_size, self.num_cells, device=values.device, dtype=values.dtype)
		grid.scatter_add_(1, idx, values)
		return grid / self.grid_counts

	def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
		flow = observations['flow'].float()
		density = observations['density'].float()
		flow_grid = self._grid_mean(flow)
		density_grid = self._grid_mean(density)

		grid = torch.stack([flow_grid, density_grid], dim=1)
		grid = grid.reshape(flow.shape[0], 2, self.grid_height, self.grid_width)

		pos_emb = torch.cat([self.x_embed(self.grid_x), self.y_embed(self.grid_y)], dim=1)
		pos_emb = pos_emb.view(self.grid_height, self.grid_width, -1).permute(2, 0, 1).unsqueeze(0)
		pos_emb = pos_emb.expand(flow.shape[0], -1, -1, -1)

		features = torch.cat([grid, pos_emb], dim=1)
		cnn_feat = self.cnn(features).mean(dim=(2, 3))
		time_feat = torch.relu(self.time_encoder(observations['time_in_day'].float().flatten(start_dim=1)))
		return torch.relu(self.proj(torch.cat([cnn_feat, time_feat], dim=1)))


class FlatDroneEncoder(nn.Module):
	def __init__(self, observation_space: spaces.Dict, hidden_dim: int = 256):
		super().__init__()
		flat_dim = int(sum(flatdim(space) for _, space in sorted(observation_space.spaces.items())))
		self.encoder = nn.Sequential(
			layer_init(nn.Linear(flat_dim, hidden_dim)),
			nn.ReLU(),
			layer_init(nn.Linear(hidden_dim, hidden_dim)),
			nn.ReLU(),
		)
		self.out_dim = hidden_dim

	def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
		flat_obs = flatten_obs_batch(observations)
		return self.encoder(flat_obs)


def build_encoder(
	observation_space: spaces.Dict,
	policy_type: str,
	map_structure: Optional[MapStructure],
	hidden_dim: int,
) -> nn.Module:
	if policy_type == 'simple':
		return SimpleDroneEncoder(observation_space, hidden_dim=hidden_dim)
	if policy_type == 'graph':
		return GraphDroneEncoder(observation_space, map_structure=map_structure, hidden_dim=hidden_dim)
	if policy_type == 'grid':
		return GridDroneEncoder(observation_space, map_structure=map_structure, hidden_dim=hidden_dim)
	return FlatDroneEncoder(observation_space, hidden_dim=hidden_dim)


class DronePPOAgent(nn.Module):
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

		self.policy_net = nn.Sequential(
			layer_init(nn.Linear(self.encoder.out_dim, policy_hidden_dim)),
			nn.ReLU(),
			layer_init(nn.Linear(policy_hidden_dim, policy_hidden_dim)),
			nn.ReLU(),
		)
		self.value_net = nn.Sequential(
			layer_init(nn.Linear(self.encoder.out_dim, value_hidden_dim)),
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
	def thunk():
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

	return thunk


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

	args.batch_size = int(args.num_envs * args.num_steps)
	if args.batch_size % args.num_minibatches != 0:
		raise ValueError('batch_size must be divisible by num_minibatches.')
	args.minibatch_size = int(args.batch_size // args.num_minibatches)
	args.num_iterations = int(args.train_timesteps // args.batch_size)
	if args.num_iterations < 1:
		raise ValueError('train-timesteps is too small for current num-envs and num-steps.')

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
	dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
	values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)

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
			rewards[step] = torch.as_tensor(reward, device=device, dtype=torch.float32)
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
					v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
				else:
					v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

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
