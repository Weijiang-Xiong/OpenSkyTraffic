"""We consider flying a fleet of drones to collect traffic data (flow, density, speed) over a large urban area.
The goal is to smartly plan the drone flights to maximize data coverage during each session.
We plan to develop an RL-based solution to this problem, which involves:
    1. A dataset `SimBarcaExplore` that provides traffic data of an urban area, where the space is divided to grids.
    2. A reward calculator based on coverage over the grid.
    3. A set of monitoring agents (drones) to query data from the dataset.
    4. An environment to orchestrate the components above.
"""
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor

from skytraffic.utils.event_logger import setup_logger
from skytraffic.utils.io import make_dir_if_not_exist
from skymonitor.simbarca_explore import SimBarcaExplore
from skymonitor.agents import BaseAgent, PolicyAgent, RandomAgent, StaticAgent
from skymonitor.policy_net import SimpleDronePolicy, GraphDronePolicy, GridDronePolicy
from skymonitor.monitor_env import build_traffic_monitor_env, TrafficMonitorEnv

POLICY_NET_DICT = {
	'simple': SimpleDronePolicy,
	'graph': GraphDronePolicy,
	'grid': GridDronePolicy,
}

class PeriodicEvalCallback(BaseCallback):

	def __init__(
		self,
		eval_env: TrafficMonitorEnv,
		num_drones: int,
		eval_freq: int,
		eval_repeat: int,
		eval_seed: int,
		verbose: int = 0,
	):
		super().__init__(verbose)
		self.eval_env = eval_env
		self.num_drones = num_drones
		self.eval_freq = int(eval_freq)
		self.eval_repeat = int(eval_repeat)
		self.eval_seed = int(eval_seed)

	def _on_step(self) -> bool:
		if self.eval_freq <= 0 or (self.n_calls % self.eval_freq) != 0:
			return True

		eval_results = defaultdict(list)
		rng = np.random.default_rng(self.eval_seed)
		seeds = rng.choice(10000, size=self.eval_repeat, replace=False)
		agent = PolicyAgent(
			num_drones=self.num_drones,
			map_structure=self.eval_env.map_structure,
			policy=self.model.policy,
		)

		for seed in seeds:
			with torch.no_grad():
				all_reward = get_reward_all_sessions(self.eval_env, agent, seed=int(seed))
			eval_results['reward'].append(all_reward.mean().item())

		stats = {}
		for key in eval_results:
			stats[key] = sum(eval_results[key]) / len(eval_results[key])
			stats['std_' + key] = np.std(eval_results[key])

		for key, value in stats.items():
			self.logger.record(f'eval/{key}', float(value))
		self.logger.dump(step=self.num_timesteps)

		return True

def train_monitoring_agent_with_ppo(
	policy_type: str = 'simple',
	total_timesteps: int = int(1e6),
	num_envs: int = 1,
	num_drones: int = 10,
	learning_rate: float = 3e-4,
	log_dir: str = 'scratch/rl_drone',
	save_name: str = "model",
	seed: int = 0,
	eval_freq: int = 0,
	eval_repeat: int = 5,
	eval_seed: int = 888,
) -> PPO:
	"""Train PPO on the monitoring environment with the custom policy."""

	set_random_seed(seed)
	log_path = Path(log_dir)
	log_path.mkdir(parents=True, exist_ok=True)
	dataset = SimBarcaExplore(split="train",norm_tid=False)

	def _make_env(rank: int):
		def _init():
			env = env = build_traffic_monitor_env(dataset=dataset, num_drones=num_drones)
			env.reset(seed=seed + rank)
			return env
		return _init

	vec_env = DummyVecEnv([_make_env(rank) for rank in range(num_envs)])
	vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10, clip_reward=10)
	vec_env = VecMonitor(vec_env)
	
	eval_env = None
	callback = None
	if eval_freq > 0:
		eval_env = build_traffic_monitor_env(dataset=SimBarcaExplore(split='test', norm_tid=False), num_drones=num_drones)
		eval_env = Monitor(eval_env)
		callback = PeriodicEvalCallback(
			eval_env=eval_env,
			num_drones=num_drones,
			eval_freq=eval_freq,
			eval_repeat=eval_repeat,
			eval_seed=eval_seed,
		)

	model = PPO(
		policy=POLICY_NET_DICT[policy_type],
		env=vec_env,
		learning_rate=learning_rate,
		n_steps=max(32, 1024 // max(1, num_envs)),
		batch_size=64,
		gamma=0.99,
		gae_lambda=0.95,
		clip_range=0.2,
		vf_coef=0.5,
		max_grad_norm=0.5,
		tensorboard_log=str(log_path),
		verbose=1,
		seed=seed,
	)

	model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callback)
	model.save(log_path / save_name)
	vec_env.close()
	if eval_env is not None:
		eval_env.close()

	return model


def get_reward_all_sessions(env: TrafficMonitorEnv, agent: BaseAgent, seed: int = 42) -> list:
	# seed the environment once before the loop to initialize its random number generator.
	# in this way the drones are randomly spawn for each session, but remains consistent across runs
	env.reset(seed=seed)  

	all_reward = []

	for active_session in range(env.total_sessions):
		# print('=== Running agents on session {} ==='.format(active_session))
		observation, info = env.reset(options={'active_session': active_session})
		done = False
		truncated = False
		episode_reward = 0.0
		while not (done or truncated):
			actions = agent.select_action(observation)
			observation, reward, done, truncated, info = env.step(actions)
			episode_reward += reward

		all_reward.append(episode_reward)

	return all_reward


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train and evaluate PPO monitoring agent.')
	parser.add_argument('--eval-only', action='store_true', help='If set, only evaluate the agent without training.')
	parser.add_argument('--agent-type', type=str, default='ppo', help='Type of agent to use: ppo, random, static.')
	parser.add_argument('--policy-type', type=str, default='simple', help='Type of policy network to use: simple, graph, grid.')
	parser.add_argument('--train-timesteps', type=int, default=int(1e6), help='Number of environment steps for PPO training.')
	parser.add_argument('--num-envs', type=int, default=8, help='Number of parallel environments during training.')
	parser.add_argument('--num-drones', type=int, default=10, help='Number of drones used in training and evaluation.')
	parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate for PPO.')
	parser.add_argument('--logdir', type=str, default='scratch/drone_monitor', help='Directory to store PPO logs and checkpoints.')
	parser.add_argument('--save-name', type=str, default="model", help='Checkpoint name to save the trained PPO agent.')
	parser.add_argument('--log-level', type=str, default='info', choices=['debug', 'info', 'warning', 'error', 'critical'], help='Logging level.')
	parser.add_argument('--train-seed', type=int, default=42, help='Random seed for training.')
	parser.add_argument('--eval-freq', type=int, default=int(5e4), help='Evaluation frequency (0 disables).')
	parser.add_argument('--eval-repeat', type=int, default=5, help='Number of evaluation runs.')
	parser.add_argument('--eval-seed', type=int, default=888, help='Seed for the evaluation environment.')
	args = parser.parse_args()

	make_dir_if_not_exist(args.logdir)
	logger = setup_logger(name='skymonitor', log_file='{}/experiment.log'.format(args.logdir), level=getattr(logging, args.log_level.upper()))
	logger.info('Arguments: {}'.format(args))

	if not args.eval_only and args.agent_type == 'ppo':
		logger.info('Training PPO monitoring agent.')
		ppo = train_monitoring_agent_with_ppo(
			policy_type=args.policy_type,
			total_timesteps=args.train_timesteps,
			num_envs=args.num_envs,
			num_drones=args.num_drones,
			learning_rate=args.learning_rate,
			log_dir=args.logdir,
			save_name=args.save_name,
			seed=args.train_seed,
			eval_freq=args.eval_freq,
			eval_repeat=args.eval_repeat,
			eval_seed=args.eval_seed,
		)

	# test the agent
	dataset = SimBarcaExplore(split='test', norm_tid=False)
	env = build_traffic_monitor_env(dataset=dataset, num_drones=args.num_drones)

	match args.agent_type:
		case 'ppo':
			if not args.eval_only:
				agent = PolicyAgent(policy=ppo)
			else:
				logger.info('Using trained PPO Agent from {}, skipping training.'.format(
					Path(args.logdir) / '{}'.format(args.save_name)
				))
				agent = PolicyAgent(policy=PPO.load(Path(args.logdir) / '{}'.format(args.save_name)))
		case 'random':
			logger.info('Using Random Agent, skipping training.')
			agent = RandomAgent(num_drones=args.num_drones)
		case _:
			logger.info('Using Static Agent, skipping training.')
			agent = StaticAgent(num_drones=args.num_drones)

	reward_reps = []
	rng = np.random.default_rng(args.eval_seed)
	seeds = rng.choice(10000, size=args.eval_repeat, replace=False)
	for i in range(args.eval_repeat):
		with torch.no_grad():
			rewards = get_reward_all_sessions(env, agent, seed=int(seeds[i]))
		reward_reps.append(rewards)

	reward_reps = np.array(reward_reps)  # (eval_repeat, total_sessions)
	stats = {
		'avg_reward': reward_reps.mean(axis=1).mean(),
		'std_reward': reward_reps.mean(axis=1).std(),
		'avg_reward_per_session': reward_reps.mean(axis=0).tolist(),
		'std_reward_per_session': reward_reps.std(axis=0).tolist(),
	}

	logger.info('Drone Monitoring Evaluation Results: {}'.format(stats))

	env.close()

	# save eval_results and stats to a json file
	save_path = Path(args.logdir) / 'multi_run_results.json'
	with open(save_path, 'w') as f:
		json.dump({'stats': stats, 'reward_all_repeats': reward_reps.tolist()}, f, indent=4)
