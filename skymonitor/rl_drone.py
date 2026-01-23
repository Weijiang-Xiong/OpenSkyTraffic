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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from skytraffic.utils.event_logger import setup_logger
from skytraffic.utils.io import make_dir_if_not_exist
from skymonitor.simbarca_explore import SimBarcaExplore
from skymonitor.agents import MonitoringAgent, DronePolicy, RandomAgent, StaticAgent
from skymonitor.monitor_env import build_monitor_env, TrafficMonitorEnv


def train_monitoring_agent_with_ppo(
	total_timesteps: int = 100,
	num_envs: int = 1,
	num_drones: int = 10,
	learning_rate: float = 3e-4,
	log_dir: str = 'scratch/rl_drone',
	checkpoint_filename: str = 'ppo.zip',
	seed: int = 0,
) -> PPO:
	"""Train PPO on the monitoring environment with the custom policy."""

	set_random_seed(seed)
	log_path = Path(log_dir)
	log_path.mkdir(parents=True, exist_ok=True)
	dataset = SimBarcaExplore(split="train",norm_tid=False)

	def _make_env(rank: int):
		def _init():
			env = env = build_monitor_env(dataset=dataset, num_drones=num_drones)
			env = Monitor(env)
			env.reset(seed=seed + rank)
			return env
		return _init

	vec_env = DummyVecEnv([_make_env(rank) for rank in range(num_envs)])

	model = PPO(
		policy=DronePolicy,
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

	model.learn(total_timesteps=total_timesteps, progress_bar=True)
	model.save(log_path / checkpoint_filename)
	vec_env.close()

	return model


def get_reward_all_sessions(env: TrafficMonitorEnv, agent: MonitoringAgent, seed: int = 42) -> torch.Tensor:
	# seed the environment once before the loop to initialize its random number generator.
	# in this way the drones are randomly spawn for each session, but remains consistent across runs
	env.reset(seed=seed)  

	all_reward = []

	for active_session in range(env.dataset.num_sessions):
		print('=== Running agents on session {} ==='.format(active_session))
		observation, info = env.reset(options={'active_session': active_session})
		done = False
		truncated = False
		episode_reward = 0.0
		while not (done or truncated):
			actions = agent.select_action(observation)
			observation, reward, done, truncated, info = env.step(actions)
			episode_reward += reward

		all_reward.append(episode_reward)

	all_reward = torch.tensor(all_reward)

	return all_reward


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train and evaluate PPO monitoring agent.')
	parser.add_argument('--agent-type', type=str, default='ppo', help='Type of agent to use: ppo, random, static.')
	parser.add_argument('--total-timesteps', type=int, default=500_000, help='Number of environment steps for PPO training.')
	parser.add_argument('--num-envs', type=int, default=8, help='Number of parallel environments during training.')
	parser.add_argument('--num-drones', type=int, default=10, help='Number of drones used in training and evaluation.')
	parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate for PPO.')
	parser.add_argument('--log-dir', type=str, default='scratch/rl_drone', help='Directory to store PPO logs and checkpoints.')
	parser.add_argument('--ckptname', type=str, default='ppo', help='Filename for the checkpoint without extension.')
	parser.add_argument('--seed', type=int, default=42, help='Random seed for training.')
	parser.add_argument('--eval-repeat', type=int, default=5, help='Number of evaluation runs.')
	parser.add_argument('--eval-env-seed', type=int, default=888, help='Seed for the evaluation environment.')
	args = parser.parse_args()

	make_dir_if_not_exist(args.log_dir)
	logger = setup_logger(name='default', log_file='{}/experiment.log'.format(args.log_dir), level=logging.INFO)

	if args.agent_type == 'ppo':
		logger.info('Training PPO monitoring agent.')
		ppo = train_monitoring_agent_with_ppo(
			total_timesteps=args.total_timesteps,
			num_envs=args.num_envs,
			num_drones=args.num_drones,
			learning_rate=args.learning_rate,
			log_dir=args.log_dir,
			checkpoint_filename='{}.zip'.format(args.ckptname),
			seed=args.seed,
		)

	# test the agent
	dataset = SimBarcaExplore(split='test', norm_tid=False)
	env = build_monitor_env(dataset=dataset, num_drones=args.num_drones)

	match args.agent_type:
		case 'ppo':
			agent = MonitoringAgent(
				num_drones=args.num_drones,
				map_structure=env.map_structure,
				policy_net=PPO.load(Path(args.log_dir) / '{}.zip'.format(args.ckptname)).policy,
			)
		case 'random':
			logger.info('Using Random Agent, skipping training.')
			agent = RandomAgent(num_drones=args.num_drones)
		case 'static':
			logger.info('Using Static Agent, skipping training.')
			agent = StaticAgent(num_drones=args.num_drones)

	eval_results = defaultdict(list)
	rng = np.random.default_rng(args.eval_env_seed)
	seeds = rng.choice(10000, size=args.eval_repeat, replace=False)
	for i in range(args.eval_repeat):
		with torch.no_grad():
			all_reward = get_reward_all_sessions(env, agent, seed=int(seeds[i]))
		eval_results['reward'].append(all_reward.mean().item())

	stats = {}
	for key in eval_results:
		stats[key] = sum(eval_results[key]) / len(eval_results[key])
		stats['std_' + key] = np.std(eval_results[key])

	logger.info('Drone Monitoring Evaluation Results: {}'.format(stats))

	env.close()

	# save eval_results and stats to a json file
	save_path = Path(args.log_dir) / 'multi_run_results_static_agent.json'
	with open(save_path, 'w') as f:
		json.dump({'eval_results': eval_results, 'stats': stats}, f, indent=4)
