"""Gymnasium environment for SkyMonitor.
see the `Gymnasium environment creation tutorial` at the link:
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation


"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib import animation
from einops import repeat

import torch

from skymonitor.simbarca_explore import SimBarcaExplore
from skymonitor.agents import RandomAgent, StaticAgent, DroneAction
from skymonitor.congestion import get_congestion_score, get_congestion_change

# make them un-mutable dataclasses for safety
@dataclass(frozen=True)
class TrafficData:
	flow: np.ndarray = None
	density: np.ndarray = None
	time_stamp: np.ndarray = None
	state: np.ndarray = None
	score: np.ndarray = None
	enters: np.ndarray = None
	exits: np.ndarray = None

# make them un-mutable dataclasses for safety
@dataclass(frozen=True)
class MapStructure:
	# the (x, y) coordinates of each node, shape (num_nodes, 2)
	node_coordinates: np.ndarray = None
	# the adjacency matrix of the grid, shape (num_nodes, num_nodes)
	adjacency_matrix: np.ndarray = None
	# the (x, y) coordinates of each node in the grid, shape (num_nodes, 2)
	grid_xy: np.ndarray = None
	# the grid ID for each node, shape (num_nodes, )
	grid_id: np.ndarray = None
	grid_xy_to_id: Dict[Tuple[int, int], int] = None
	available_positions: List[Tuple[int, int]] = None

	def __post_init__(self):
		if self.available_positions is None:
			positions = [] if self.grid_xy_to_id is None else list(self.grid_xy_to_id.keys())
			object.__setattr__(self, 'available_positions', positions)
		if self.grid_xy_to_id is None and self.grid_xy is not None and self.grid_id is not None:
			grid_xy_to_id = {
				(x, y): gid for (x, y), gid in zip(self.grid_xy.tolist(), self.grid_id.tolist())
			}
			object.__setattr__(self, 'grid_xy_to_id', grid_xy_to_id)


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
		data: TrafficData,
		map_structure: MapStructure,
		num_drones: int,
		seed: int = None,
	):
		super().__init__()
		# data container
		self.traffic_data = data
		# map structure info
		self.map_structure = map_structure
		# settings and configurations
		self.num_drones = int(num_drones)
		# the random seed to be used upon next reset, in fact, each active session can be considered as a separate environment
		# we want each reset to use a different active session and initial locations, so the agent can explore this dataset better
		self.next_reset_seed: int = seed
		# spaces of the environment
		self.action_space = spaces.MultiDiscrete([len(DroneAction)] * num_drones)
		S, T, P = self.traffic_data.flow.shape
		self.observation_space = spaces.Dict(
			{
				# observed traffic data and time info
				'flow': spaces.Box(low=0.0, high=2.0, shape=(P, ), dtype=np.float32),
				'density': spaces.Box(low=0.0, high=2.0, shape=(P, ), dtype=np.float32),
				'time_in_day': spaces.Box(low=0.0, high=1.0, shape=(1, ), dtype=np.float32),
				# the coordinates of drones and the node coverage mask
				'positions_x': spaces.MultiDiscrete(
					[int(self.map_structure.grid_xy[:, 0].max()) + 1] * self.num_drones
				),
				'positions_y': spaces.MultiDiscrete(
					[int(self.map_structure.grid_xy[:, 1].max()) + 1] * self.num_drones
				),
				'coverage_mask': spaces.MultiBinary(n=P),
			}
		)
		self.total_sessions = S
		self.total_steps = T
		self.total_locations = P

		# empty grid cells are not valid positions
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
		self._active_session = None

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
			active_session = self.np_random.integers(0, self.total_sessions).item()
		self.active_session = active_session

		self.data_iterator = self.session_data_iterator()
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
		# this is never reached normally since we check the termination flag with total_steps
		except StopIteration:
			return dict(), 0.0, True, False, {}

		pos = self.apply_actions(actions)
		obs = self.get_observations(pos)

		# the reward is calculated based on the current observation
		reward_value = self.calculate_reward(obs)

		terminated_flag = self.step_index >= (self.total_steps - 1)
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

	@property
	def active_session(self):
		return self._active_session

	@active_session.setter
	def active_session(self, idx):
		try:
			idx = int(idx)
		except Exception as e:
			raise ValueError(f"Active session should be an integer: {e}")
		self._active_session = idx % self.total_steps

	def session_data_iterator(self):
		for i in range(self.total_steps):
			sample = {
				'flow': self.traffic_data.flow[self.active_session, i, :],
				'density': self.traffic_data.density[self.active_session, i, :],
				# ensure the shape is (1, ), match the observation space definition
				'time_in_day': np.atleast_1d(self.traffic_data.time_stamp[i]).astype(np.float32),
			}
			yield i, sample

	def calculate_reward(self, observation: Dict[str, np.ndarray]) -> float:
		"""Define the reward based on the current observation."""
		coverage = observation['coverage_mask']
		covered_enters = (coverage.squeeze() * self.traffic_data.enters[self.active_session, self.step_index, :]).sum()
		covered_exits = (coverage.squeeze() * self.traffic_data.exits[self.active_session, self.step_index, :]).sum()

		return float((covered_enters + covered_exits).item())

	def init_agent_positions(self):
		"""Initialize drone positions at start of episode."""
		all_positions = set(self.map_structure.available_positions)
		interior_positions = [
			(x, y) for x, y in all_positions
			if (x + 1, y) in all_positions
			and (x - 1, y) in all_positions
			and (x, y + 1) in all_positions
			and (x, y - 1) in all_positions
		]
		indices = self.np_random.choice(len(interior_positions), size=self.num_drones, replace=False)
		return [interior_positions[idx] for idx in indices]

	def compute_coverage_mask(self, positions: List[Tuple[int, int]]) -> np.ndarray:
		mask = sum(
			[
				self.map_structure.grid_id == self.map_structure.grid_xy_to_id[(x, y)]
				for x, y in positions
			]
		)
		return (mask > 0).astype(np.int8)

	def get_observations(self, positions: List[Tuple[int, int]]) -> Dict[str, np.ndarray]:
		"""get the observed traffic at the given positions."""

		# clean data from the dataset, for 1 time step
		observed_traffic = self.data_sample
		coverage_mask = self.compute_coverage_mask(positions)
		positions_x = np.array([pos[0] for pos in positions], dtype=np.int64)
		positions_y = np.array([pos[1] for pos in positions], dtype=np.int64)

		# now we modify the nan values to be 0 to align with the observation space definition
		for key, value in observed_traffic.items():
			if not isinstance(value, np.ndarray):
				continue
			if value.shape != coverage_mask.shape:
				continue
			observed_traffic[key] = np.nan_to_num(value, nan=0.0)
			observed_traffic[key][coverage_mask == 0] = 0.0

		observation = {
			**observed_traffic,
			'coverage_mask': coverage_mask,
			'positions_x': positions_x,
			'positions_y': positions_y,
		}

		return observation

	def get_info(self) -> Dict:
		return {
			'step_index': self.step_index,
			'session_index': self.active_session,
			'positions': list(self.positions),
			'state': self.traffic_data.state[self.active_session, self.step_index, :],
			'congestion_score': self.traffic_data.score[self.active_session, self.step_index, :],
			'enters': self.traffic_data.enters[self.active_session, self.step_index, :],
			'exits': self.traffic_data.exits[self.active_session, self.step_index, :],
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
		if new_pos not in self.map_structure.available_positions:
			new_pos = old_pos
		return new_pos

	def render(self):
		pass

	def visualize_traj(self, trajectory: List):
		"""Visualize drone paths with matplotlib in terminal mode."""

		positions_frames = [np.asarray(frame) for frame in trajectory]
		steps = len(positions_frames)

		grid_xy = self.map_structure.grid_xy
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
		for (x, y), label in zip(grid_xy, self.map_structure.grid_id):
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


def build_monitor_env(
	dataset: SimBarcaExplore,
	num_drones=10,
	density_threshold=0.001,
	ff_quantile=0.9,
	entry_thd=0.5,
	exit_thd=0.6,
	seed=42,
):
	# the time window and time step setting are not relevant to RL environment
	# as they are related to generating indexes for data sampling batch-wise training
	# in RL env, we take the data sequences directly and move 1 step each time, so no need for the complication
	# norm_tid=True normalizes the time in day encoding [8am, 10am] to have zero mean and unit variance
	flow, density = dataset.veh_flow_3min, dataset.veh_density_3min

	congestion_score = get_congestion_score(
		flow=flow,
		density=density,
		density_threshold=density_threshold,
		ff_quantile=ff_quantile,
	)
	state, enters, exits = get_congestion_change(congestion_score, entry_thd=entry_thd, exit_thd=exit_thd)

	traffic_data = TrafficData(
		flow=flow,
		density=density,
		time_stamp=dataset.time_in_day_3min,
		state=state,
		score=congestion_score,
		enters=enters,
		exits=exits,
	)
	map_structure = MapStructure(
		node_coordinates=dataset.node_coordinates,
		adjacency_matrix=dataset.adjacency,
		grid_xy=dataset.grid_xy,
		grid_id=dataset.grid_id,
		grid_xy_to_id=dataset.grid_xy_to_id,
	)

	env = TrafficMonitorEnv(
		data=traffic_data,
		map_structure=map_structure,
		num_drones=num_drones,
		seed=seed,
	)

	return env


if __name__ == '__main__':
	from skytraffic.utils.event_logger import setup_logger
	from stable_baselines3.common.env_checker import check_env

	logger = setup_logger(name='default', log_file='./scratch/drone_monitor_env.log', level=logging.INFO)

	num_drones = 10
	
	dataset = SimBarcaExplore(split="train", norm_tid=False)
	env = build_monitor_env(dataset=dataset, num_drones=num_drones)

	check_env(env, warn=True)

	agent = StaticAgent(num_drones=num_drones)

	observation, info = env.reset()
	logger.info('Initial observation keys:{}'.format(observation.keys()))
	logger.info('Initial coverage ratio: {}'.format(observation['coverage_mask'].mean()))
	terminated = False
	while not terminated:
		obs, reward_value, terminated, _, info = env.step(agent.select_action(observation))  # test step
		logger.info(
			'Step:{} Reward:{:.4f} Flow:{:.4f} Density:{:.4f} TimeInDay:{:.4f} Coverage:{:.4f}'.format(
				info['step_index'],
				reward_value,
				obs['flow'].mean(),
				obs['density'].mean(),
				obs['time_in_day'].mean(),
				obs['coverage_mask'].mean(),
			)
		)
	
	animation_obj = env.visualize_traj(env.positions_history)
	animation_obj.save("./figures/skymonitor/example_traj.gif", writer=animation.PillowWriter(fps=2))
	print("Saved example trajectory visualization to ./figures/skymonitor/example_traj.gif")
