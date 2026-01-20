"""Gymnasium environment for SkyMonitor.
see the `Gymnasium environment creation tutorial` at the link:
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation


"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib import animation
from einops import repeat

import torch

from skymonitor.simbarca_explore import SimBarcaExplore
from skymonitor.agents import RandomAgent, DroneAction


class TrafficMonitorEnv(gym.Env):
	"""
	Design principles for the environment (following Gymnasium API):
	- **🎯 Agent Skill**: Collect traffic data in a grid map
	- **👀 Information**: Observed traffic states, agent position, map structure
	- **🎮 Actions**: Move up, down, left or right by 1 or 2 steps; Move diagonally by 1 step; Don't move
	- **🏆 Success**: maximize the defined reward
	- **⏰ End**: A simulation session ends.
	"""

	metadata = {'render_modes': ['human'], 'name': 'drone_traffic_v0', 'render_fps': 30}

	def __init__(
		self,
		dataset: SimBarcaExplore,
		num_drones: int,
		seed: int = None,
	):
		super().__init__()
		# components of the environment
		self.dataset = dataset
		self.action_space = spaces.MultiDiscrete([len(DroneAction)] * num_drones)
		# settings and configurations
		self.num_drones = int(num_drones)
		# the random seed to be used upon next reset, in fact, each active session can be considered as a separate environment
		# we want each reset to use a different active session and initial locations, so the agent can explore this dataset better
		self.next_reset_seed: int = seed

		indata_shape: Tuple = dataset.input_size
		cvg_mask_shape: int = dataset.input_size[:-1]  # excluding the feature dim

		self.observation_space = spaces.Dict(
			{
				'observed_traffic': spaces.Box(
					low=0.0,
					high=np.inf,
					shape=indata_shape,
					dtype=np.float32,
				),
				'coverage_mask': spaces.MultiBinary(n=cvg_mask_shape),
			}
		)

		# data shape info
		in_data_steps, num_locations, feature_dim = dataset.input_size
		# we move our drones every `time_step` (3 min), and the input drone data are given every data_step (5s)
		self.in_time_steps = int(dataset.input_window // dataset.step_size)
		self.in_data_steps = int(in_data_steps)
		self.data_pt_per_time_step = int(self.in_data_steps // self.in_time_steps)
		self.num_locations = int(num_locations)
		self.feature_dim = int(feature_dim)
		# -1 because we need 1 sample for initial observation in reset()
		self.max_steps_per_session = dataset.sample_per_session - 1

		# grid structure info
		self.grid_ids: np.ndarray = dataset.grid_id
		self.grid_xy_to_id: Dict[Tuple[int, int], int] = dataset.grid_xy_to_id
		# empty grid cells are not valid positions
		self.available_positions = list(self.grid_xy_to_id.keys())
		self._action_to_direction = {
			DroneAction.STAY: (0, 0),
			DroneAction.UP: (0, 1),
			DroneAction.DOWN: (0, -1),
			DroneAction.LEFT: (-1, 0),
			DroneAction.RIGHT: (1, 0),
		}

		# running states
		self.step_index: int = 0
		self.data_sample: Dict[str, torch.Tensor] = None
		self.positions: List[Tuple[int, int]] = []
		self.positions_history = list()
		self.observation_history: List[Dict[str, torch.Tensor]] = list()
		self._last_animation = None

	def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
		"""Resets environment and returns observations and infos for all agents."""
		self.clear_state()
		# each time we reset the environment, we use a new random seed because many things, including
		# the drone locations, are generated using this seed, and we want the agent to explore
		# different initial positions rather than starting always from the same ones.
		super().reset(seed=seed if seed is not None else self.next_reset_seed)
		print(f'Environment reset with seed: {self.np_random_seed}')
		self.next_reset_seed = self.np_random.integers(100_000_000, 999_999_999).item()

		# accept override from options if provided, otherwise randomly select active session(s)
		active_session = None if not isinstance(options, dict) else options.get('active_session', None)
		if active_session is None:
			active_session = self.np_random.integers(0, self.dataset.num_sessions)
		self.dataset.active_session = active_session

		# self.compute_congestion_change()

		self.data_iterator = self.dataset.iterate_active_session()
		self.step_index, self.data_sample = next(self.data_iterator)
		assert self.step_index == 0, 'Step index should start from 0 after reset.'

		positions = self.init_agent_positions()
		obs = self.get_observations(positions)

		self.update_history(positions, obs)
		info = self.get_info()

		return obs, info


	def step(self, actions: List[DroneAction]) -> Tuple[Dict, float, bool, bool, Dict]:
		"""In the previous step, the agent took `actions` to move the drones to new positions.
		The reward is computed from the new observation.
		"""
		try:
			self.step_index, self.data_sample = next(self.data_iterator)
		# this is never reached normally since we check the termination flag with max_steps_per_session
		except StopIteration:
			return dict(), 0.0, True, False, {}

		pos = self.apply_actions(actions)
		obs = self.get_observations(pos)

		# the reward is calculated based on the current observation
		reward_value = self.calculate_reward(obs)

		terminated_flag = self.step_index >= (self.max_steps_per_session - 1)
		truncated_flag = False

		self.update_history(pos, obs)
		info = self.get_info()

		return obs, reward_value, terminated_flag, truncated_flag, info

	def close(self):
		self.clear_state()

	def clear_state(self) -> None:
		self.step_index = 0
		self.data_sample = None
		self._last_animation = None
		self.positions.clear()
		self.positions_history.clear()
		self.observation_history.clear()

	def calculate_reward(self, observation: Dict[str, np.ndarray]) -> float:
		"""Define the reward based on the current observation."""
		coverage = observation['coverage_mask']
		return float(coverage.mean())

	def init_agent_positions(self):
		"""Initialize drone positions at start of episode."""
		indices = self.np_random.choice(len(self.available_positions), size=self.num_drones, replace=False)
		return [self.available_positions[idx] for idx in indices]

	def compute_coverage_mask(self, positions: List[Tuple[int, int]]) -> np.ndarray:
		return self.dataset._compute_coverage_mask(positions)

	def compute_congestion_change(self) -> None:
		raise NotImplementedError('Define congestion change computation here.') 

	def get_observations(self, positions: List[Tuple[int, int]]) -> Dict[str, np.ndarray]:
		"""get the observed traffic at the given positions."""

		# clean data from the dataset, for 1 time step
		# shape (in_steps, num_locations, feature_dim)
		observed_traffic = self.data_sample['source'].detach().cpu().clone().numpy()
		coverage_mask = self.compute_coverage_mask(positions)  # shape (num_locations,)
		# have the same shape as observed_traffic for easier masking
		coverage_mask = repeat(coverage_mask, 'P -> T P', T=observed_traffic.shape[0])

		# now we modify the nan values to be 0 to align with the observation space definition
		observed_traffic = np.nan_to_num(observed_traffic, nan=0.0)
		observed_traffic[..., coverage_mask == 0, :-1] = 0.0

		observation = {
			'observed_traffic': observed_traffic,
			'coverage_mask': coverage_mask,
		}

		return observation

	def get_info(self) -> Dict:
		return {
			'positions': list(self.positions),
			'step_index': self.step_index,
			'session_index': self.dataset.active_session,
		}

	def update_history(self, positions: List[Tuple[int, int]], observation: Dict[str, np.ndarray]) -> None:
		self.positions = positions
		self.positions_history.append(tuple(self.positions))
		self.observation_history.append(observation)

		assert len(self.observation_history) == (self.step_index + 1), 'Observation history length mismatch.'

	def apply_actions(self, actions: np.array) -> List[Tuple[int, int]]:
		def _apply_single_env(actions: List[DroneAction], current_pos: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
			new_positions: List[Tuple[int, int]] = []
			occupied = set()
			for idx, act in enumerate(actions):
				pos = self._get_new_position(act, current_pos[idx])
				if pos in occupied:  # stay if the new position is already occupied
					pos = current_pos[idx]
				new_positions.append(pos)
				occupied.add(pos)
			return new_positions

		def _to_act_space(arr: np.ndarray) -> List[DroneAction]:
			return [DroneAction(a) for a in np.asarray(arr).tolist()]

		return _apply_single_env(_to_act_space(actions), self.positions)

	def _get_new_position(self, action: DroneAction, old_pos: Tuple[int, int]) -> Tuple[int, int]:
		"""works for 1 single position update"""
		direction = self._action_to_direction.get(action, (0, 0))
		new_pos = (old_pos[0] + direction[0], old_pos[1] + direction[1])
		if new_pos not in self.available_positions:
			new_pos = old_pos
		return new_pos

	def render(self):
		pass

	def visualize_traj(self, trajectory: List):
		"""Visualize drone paths with matplotlib in terminal mode."""

		positions_frames = [np.asarray(frame) for frame in trajectory]
		steps = len(positions_frames)

		grid_xy = self.dataset.grid_xy
		grid_limits = (
			int(grid_xy[:, 0].max() + 1),
			int(grid_xy[:, 1].max() + 1),
		)

		fig, ax = plt.subplots(figsize=(12, 8))
		ax.set_xlim(-0.5, grid_limits[0] - 0.5)
		ax.set_ylim(-0.5, grid_limits[1] - 0.5)
		for x in range(grid_limits[0] + 1):
			ax.axvline(x - 0.5, color='gray', linewidth=0.5, alpha=0.4)
		for y in range(grid_limits[1] + 1):
			ax.axhline(y - 0.5, color='gray', linewidth=0.5, alpha=0.4)
		ax.set_title('Drone Coverage Summary')
		ax.set_xlabel('Grid X')
		ax.set_ylabel('Grid Y')

		# initialize empty scatter plots, trails and annotations.
		scatter = ax.scatter([], [], c='tab:red', s=80, label='Drones')
		trails = [ax.plot([], [], linestyle='-', marker='o', alpha=0.5)[0] for _ in range(self.num_drones)]
		for (x, y), label in zip(grid_xy, self.dataset.grid_id):
			ax.text(
				x,
				y,
				str(label),
				ha='center',
				va='center',
				fontsize=8,
				color='black',
			)
		annotation = ax.text(
			0.02,
			0.95,
			'',
			transform=ax.transAxes,
			fontsize=10,
			ha='left',
			va='top',
			bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.7},
		)

		def init():
			annotation.set_text('')
			for line in trails:
				line.set_data([], [])
			return [scatter, annotation, *trails]

		def update(frame_idx):
			scatter.set_offsets(positions_frames[frame_idx])
			annotation.set_text(f'Step {frame_idx} / {steps - 1}')
			for drone_idx, line in enumerate(trails):
				history = np.asarray([frame[drone_idx] for frame in positions_frames[: frame_idx + 1]])
				line.set_data(history[:, 0], history[:, 1])
			return [scatter, annotation, *trails]

		base_interval = 1000 / max(1, self.metadata.get('render_fps', 30))
		interval_ms = max(350, base_interval)
		animation_obj = animation.FuncAnimation(
			fig,
			update,
			frames=steps,
			init_func=init,
			blit=False,
			interval=interval_ms,
			repeat=False,
		)
		self._last_animation = animation_obj

		return animation_obj


if __name__ == '__main__':
	from skytraffic.utils.event_logger import setup_logger
	from stable_baselines3.common.env_checker import check_env

	logger = setup_logger(name='default', log_file='./scratch/drone_monitor_env.log', level=logging.INFO)

	num_drones = 10
	dataset = SimBarcaExplore(
		split='train',
		input_window=3,
		pred_window=30,
		step_size=3,
		norm_tid=False,
	)
	env = TrafficMonitorEnv(
		dataset=dataset,
		num_drones=num_drones,
	)
	agent = RandomAgent(num_drones=num_drones)

	observation, info = env.reset()
	logger.info('Initial observation keys:{}'.format(observation.keys()))
	logger.info('Initial coverage ratio: {}'.format(observation['coverage_mask'].mean()))
	terminated = False
	while not terminated:
		obs, reward_value, terminated, _, info = env.step(agent.select_action(observation))  # test step
		logger.info(
			'Step:{} Reward:{:.4f} Coverage:{:.4f}'.format(
				info['step_index'],
				reward_value,
				obs['coverage_mask'].mean(),
			)
		)

	check_env(env, warn=True)
