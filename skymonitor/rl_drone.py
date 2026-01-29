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
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor

from skytraffic.utils.event_logger import setup_logger
from skytraffic.utils.io import make_dir_if_not_exist
from skymonitor.agents import PolicyAgent, RandomAgent, StaticAgent
from skymonitor.policy_net import SimpleDronePolicy, GraphDronePolicy, GridDronePolicy
from skymonitor.monitor_env import build_traffic_monitor_env, TrafficMonitorEnv, eval_on_all_sessions
from skymonitor.simbarca_explore import initialize_dataset

POLICY_NETS = {
	'simple': SimpleDronePolicy,
	'graph': GraphDronePolicy,
	'grid': GridDronePolicy,
}

class PeriodicEvalCallback(BaseCallback):

	def __init__(
		self,
		eval_env: TrafficMonitorEnv,
		policy: PPO,
		eval_freq: int,
		eval_repeat: int,
		eval_seed: int,
		verbose: int = 0,
	):
		super().__init__(verbose)
		self.eval_env = eval_env
		self.agent = PolicyAgent(policy=policy)
		self.eval_freq = int(eval_freq)
		self.eval_repeat = int(eval_repeat)
		self.eval_seed = int(eval_seed)

	def _on_step(self) -> bool:
		if self.eval_freq <= 0 or (self.n_calls % self.eval_freq) != 0:
			return True

		eval_results = defaultdict(list)
		rng = np.random.default_rng(self.eval_seed)
		seeds = rng.choice(10000, size=self.eval_repeat, replace=False)

		for seed in seeds:
			with torch.no_grad():
				eval_res = eval_on_all_sessions(self.eval_env, self.agent, seed=int(seed))
			eval_results['reward'].append(np.mean(eval_res['all_reward']))

		stats = {}
		for key in eval_results:
			stats[key] = sum(eval_results[key]) / len(eval_results[key])
			stats['std_' + key] = np.std(eval_results[key])

		for key, value in stats.items():
			self.logger.record(f'eval/{key}', float(value))
		self.logger.dump(step=self.num_timesteps)

		return True
	
	def _on_training_end(self):
		self.eval_env.close()
		return super()._on_training_end()

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
	norm_obs: bool = False,
) -> PPO:
	"""Train PPO on the monitoring environment with the custom policy."""

	set_random_seed(seed)
	log_path = Path(log_dir)
	log_path.mkdir(parents=True, exist_ok=True)

	logger = logging.getLogger('skymonitor.train_ppo')

	trainset, testset = initialize_dataset()

	def _make_env(rank: int):
		def _init():
			logger.info(f'Creating environment {rank} with seed {seed + rank}')
			env = build_traffic_monitor_env(trainset=trainset, testset=None, num_drones=num_drones, env_type='train', norm_obs=norm_obs)
			env.reset(seed=seed + rank)
			return env
		return _init

	vec_env = DummyVecEnv([_make_env(rank) for rank in range(num_envs)])
	# don't normalize the observation in VecNormalize, as the normalization is handled by the environment
	vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_obs=10, clip_reward=10)
	vec_env = VecMonitor(vec_env)

	policy_kwargs = {
		'map_structure': vec_env.get_attr('map_structure')[0],
	}

	logger.info(f'Initializing PPO model. Using policy network: {POLICY_NETS[policy_type].__name__}')
	model = PPO(
		policy=POLICY_NETS[policy_type],
		env=vec_env,
		policy_kwargs=policy_kwargs,
		learning_rate=learning_rate,
		n_steps=max(32, 2048 // max(1, num_envs)),
		batch_size=128,
		gamma=0.99,
		gae_lambda=0.95,
		clip_range=0.2,
		vf_coef=0.5,
		max_grad_norm=0.5,
		tensorboard_log=str(log_path),
		verbose=1,
		seed=seed,
	)

	logger.info('Starting PPO training with periodic evaluation callback.')
	callback = PeriodicEvalCallback(
		eval_env=build_traffic_monitor_env(trainset=trainset, testset=testset, num_drones=num_drones, env_type='test', norm_obs=norm_obs),
		policy=model,
		eval_freq=eval_freq,
		eval_repeat=eval_repeat,
		eval_seed=eval_seed,
	)

	model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callback)
	model.save(log_path / save_name)
	logger.info(f'Trained PPO model saved to {log_path / save_name}')

	vec_env.close()

	return model


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train and evaluate PPO monitoring agent.')
	parser.add_argument('--eval-only', action='store_true', help='If set, only evaluate the agent without training.')
	parser.add_argument('--agent-type', type=str, default='ppo', help='Type of agent to use: ppo, random, static.')
	parser.add_argument('--policy-type', type=str, default='simple', help='Type of policy network to use: simple, graph, grid.')
	parser.add_argument('--train-timesteps', type=int, default=int(1e6), help='Number of environment steps for PPO training.')
	parser.add_argument('--num-envs', type=int, default=8, help='Number of parallel environments during training.')
	parser.add_argument('--num-drones', type=int, default=10, help='Number of drones used in training and evaluation.')
	parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate for PPO.')
	parser.add_argument('--logdir', type=str, default='scratch/drone_monitor', help='Directory to store PPO logs and checkpoints.')
	parser.add_argument('--save-name', type=str, default="model", help='Checkpoint name to save the trained PPO agent.')
	parser.add_argument('--log-level', type=str, default='info', choices=['debug', 'info', 'warning', 'error', 'critical'], help='Logging level.')
	parser.add_argument('--train-seed', type=int, default=42, help='Random seed for training.')
	parser.add_argument('--eval-freq', type=int, default=int(1e4), help='Evaluation frequency (0 disables). Will run every `eval-freq * num_envs` time steps')
	parser.add_argument('--eval-repeat', type=int, default=5, help='Number of evaluation runs.')
	parser.add_argument('--eval-seed', type=int, default=888, help='Seed for the evaluation environment.')
	parser.add_argument('--norm-obs', action=argparse.BooleanOptionalAction, default=True, help='Normalize observations, default True. Pass --no-norm-obs to disable.')
	args = parser.parse_args()

	make_dir_if_not_exist(args.logdir)
	logger = setup_logger(
		name='skymonitor',
		log_file='{}/experiment.log'.format(args.logdir),
		level=getattr(logging, args.log_level.upper()),
	)
	logger.info('Arguments: {}'.format(args))

	if not args.eval_only and args.agent_type == 'ppo':
		logger.info('Training PPO monitoring agent.')
		trained_ppo = train_monitoring_agent_with_ppo(
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
			norm_obs=args.norm_obs,
		)

	# test the agent
	logger.info('Evaluating the {} monitoring agent.'.format(args.agent_type))
	trainset, testset = initialize_dataset()
	env: TrafficMonitorEnv = build_traffic_monitor_env(
		trainset=trainset, testset=testset, num_drones=args.num_drones, env_type='test', norm_obs=args.norm_obs
	)

	match args.agent_type:
		case 'ppo':
			if not args.eval_only:
				agent = PolicyAgent(policy=trained_ppo)
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

	res_reps = defaultdict(list)
	rng = np.random.default_rng(args.eval_seed)
	seeds = rng.choice(10000, size=args.eval_repeat, replace=False)
	for i in range(args.eval_repeat):
		with torch.no_grad():
			eval_res = eval_on_all_sessions(env, agent, seed=int(seeds[i]))
		res_reps['reward'].append(eval_res['all_reward'])
		res_reps['trajectories'].append(eval_res['all_trajectories'])

	reward_reps = np.array(res_reps['reward'])  # (eval_repeat, total_sessions)
	stats = {
		'avg_reward': reward_reps.mean(axis=1).mean(),
		'std_reward': reward_reps.mean(axis=1).std(),
		'avg_reward_per_session': reward_reps.mean(axis=0).tolist(),
		'std_reward_per_session': reward_reps.std(axis=0).tolist(),
	}

	logger.info('Drone Monitoring Evaluation Results: {}'.format(stats))

	# save eval_results and stats to a json file
	save_path = Path(args.logdir) / 'rep{}x_results.json'.format(args.eval_repeat)
	with open(save_path, 'w') as f:
		json.dump({'stats': stats, 'reward_all_repeats': reward_reps.tolist(), 'trajectories_all_repeats': res_reps['trajectories']}, f, indent=4)

	logger.info("Visualizeing the trajectories of some evaluation runs and sessions.")
	for rep, all_traj in enumerate(res_reps['trajectories']):
		if not rep % 5 == 0:
			continue
		for session_id, traj in enumerate(all_traj):
			if session_id % 20 == 0:
				env.visualize_traj(
					traj,
					save_path=Path(args.logdir) / f'traj_rep{rep}_session{session_id}.gif',
				)

	env.close()
	logger.info('Evaluation completed.')
