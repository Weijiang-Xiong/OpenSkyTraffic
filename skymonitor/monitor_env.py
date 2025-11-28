"""Gymnasium environment for SkyMonitor.
see the `Gymnasium environment creation tutorial` at the link:
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation


"""
from copy import deepcopy
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib import animation

import torch

from skymonitor.simbarca_explore import SimBarcaExplore
from skymonitor.agents import RandomAgent, DroneAction
from skymonitor.traffic_predictor import TrafficPredictor

from stable_baselines3.common.vec_env import VecEnv


class TrafficMonitorEnv(gym.Env):
	"""
	Design principles for the environment (following Gymnasium API):
	- **🎯 Agent Skill**: Collect traffic data in a grid map
	- **👀 Information**: Predicted traffic states, agent position, map structure
	- **🎮 Actions**: Move up, down, left or right by 1 or 2 steps; Move diagonally by 1 step; Don't move
	- **🏆 Success**: the predictor has a smaller error compared to a naive flight plan
	- **⏰ End**: A simulation session ends.

	Note on `TrafficMonitorEnv`'s vectorized mode:
		0. This mode DOES NOT work with stable-baselines3's PPO implementation, use the wrapper `TrafficMonitorEnvVectorized` instead.
	    1. The environment can be instantiated in vectorized mode by providing `num_vec_env` > 1.
	    2. This mode literally retrieves multiple samples from different simulation sessions, but these samples
	    have the same time index (e.g., all of them have input from 8:00 to 8:30 and predict from 8:30 to 9:00),
		and they will terminate at the same time. 
	    3. the single-env action space and observation spaces are kept unchanged to let PPO work correctly with its batching.
	    4. the dataset have a tensor as its `active_session` to indicate which session is active for each vector env.
	    5. the predictor now deals with batch_size = num_vec_env (in non-vectorized mode, batch_size = 1)
	"""

	metadata = {'render_modes': ['human'], 'name': 'drone_traffic_v0', 'render_fps': 30}

	def __init__(
		self,
		dataset: SimBarcaExplore,
		predictor: TrafficPredictor,
		num_drones: int,
		num_vec_env: int = None,
		seed: int = None,
	):
		super().__init__()
		# components of the environment
		self.dataset = dataset
		self.predictor = predictor
		self.action_space = spaces.MultiDiscrete([len(DroneAction)] * num_drones)
		self.baseline_agent = RandomAgent(action_space=self.action_space)
		# settings and configurations
		self.num_vec_env = num_vec_env
		self.num_drones = int(num_drones)
		# the random seed to be used upon next reset, in fact, each active session can be considered as a separate environment
		# we want each reset to use a different active session and initial locations, so the agent can explore this dataset better
		self.next_reset_seed: int = seed

		indata_shape: Tuple = dataset.input_size
		pred_shape: Tuple = dataset.output_size
		cvg_mask_shape: int = dataset.adjacency.shape[0]

		self.observation_space = spaces.Dict(
			{
				'observed_traffic': spaces.Box(
					low=0.0,
					high=np.inf,
					shape=indata_shape,
					dtype=np.float32,
				),
				'coverage_mask': spaces.MultiBinary(n=cvg_mask_shape),
				'batch_pred': spaces.Box(
					low=-np.inf,
					high=np.inf,
					shape=pred_shape,
					dtype=np.float32,
				),
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
		self.predictor_input_steps = int(predictor.in_steps)
		self.future_horizon = int(predictor.output_steps)
		self.max_steps_per_session = dataset.sample_per_session

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

	@property
	def is_vectorized(self) -> bool:
		return self.num_vec_env is not None

	def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
		"""Resets environment and predictor.
		Returns observations and infos for all agents.
		"""
		self.clear_state()
		# each time we reset the environment, we use a new random seed because many things, including
		# the drone locations, are generated using this seed, and we want the agent to explore 
		# different initial positions rather than starting always from the same ones.
		super().reset(seed=seed if seed is not None else self.next_reset_seed)
		print(f"Environment reset with seed: {self.np_random_seed}")
		self.next_reset_seed = self.np_random.integers(100_000_000, 999_999_999).item()

		# accept override from options if provided, otherwise randomly select active session(s)
		options = dict() if options is None else options
		self.set_dataset_active_session(options.get('active_session', None))

		self.data_iterator = self.dataset.iterate_active_session()
		self.step_index, self.data_sample = next(self.data_iterator)
		assert self.step_index == 0, 'Step index should start from 0 after reset.'

		positions = self.init_agent_positions()
		obs = self.get_traffic_obs_pred(positions)

		self.baseline_agent = RandomAgent(action_space=self.action_space)
		self.b_actions = self.baseline_agent.select_action(obs)

		self.update_history(positions, obs)
		info = self.get_info()

		return obs, info

	def set_dataset_active_session(self, active_session_option=None) -> None:
		active_session = active_session_option if active_session_option is not None else None
		if active_session is None:
			if not self.is_vectorized:
				active_session = self.np_random.integers(0, self.dataset.num_sessions)
			else:
				active_session = self.np_random.integers(0, self.dataset.num_sessions, size=self.num_vec_env)
		self.dataset.active_session = active_session

	def step(self, actions: List[DroneAction]) -> Tuple[Dict, float, bool, bool, Dict]:
		"""In the previous step, the agent took `actions` to move the drones to new positions.
		The predictor makes prediction based on the traffic of the new positions, and we compare
		it with the baseline agent's prediction to compute the reward.
		"""
		try:
			self.step_index, self.data_sample = next(self.data_iterator)
		# this is never reached normally since we check the termination flag with max_steps_per_session
		except StopIteration:  
			return dict(), 0.0, True, False, {}

		pos = self.apply_actions(actions)
		obs = self.get_traffic_obs_pred(pos)

		# the positions chosen by the baseline agent (from last step observations)
		b_positions = self.apply_actions(self.b_actions)
		# the resulting observation and prediction
		b_obs = self.get_traffic_obs_pred(b_positions)

		# the reward is calculated based on the prediction advantage over the baseline's choices
		reward_value = self.calculate_reward(
			torch.from_numpy(obs['batch_pred']),
			torch.from_numpy(b_obs['batch_pred']),
			self.data_sample['target'],
		)

		terminated_flag = self.step_index >= (self.max_steps_per_session - 1)
		truncated_flag = False

		# update the baseline agent's action for next step
		self.b_actions = self.baseline_agent.select_action(obs)

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

	def calculate_reward(self, pred: torch.Tensor, b_pred: torch.Tensor, gt: torch.Tensor) -> float | List[float]:
		"""The agent will be rewarded primariy based on improvements of prediction quality:
		    ** The predictor achieves a smaller error compared to a baseline agent (e.g., random flight plan) **
		And in addition, we add rewards to encourage:
		    1. larger coverage
		    2. ???
		"""
		if not self.is_vectorized:
			pred_error = torch.mean((pred - gt) ** 2).item()
			b_pred_error = torch.mean((b_pred - gt) ** 2).item()
			return 1.0 if (pred_error - b_pred_error) > 0 else 0.0

		else: # return the per-env reward as a list of floats
			pred_error = torch.mean((pred - gt) ** 2, dim=tuple(range(1, pred.ndim))) # (N, )
			b_pred_error = torch.mean((b_pred - gt) ** 2, dim=tuple(range(1, b_pred.ndim)))  # (N, )
			return ((b_pred_error - pred_error) > 0).cpu().numpy().tolist()
		

	def init_agent_positions(self):
		"""Initialize drone positions at start of episode."""

		def _init_agent_positions():
			"""non-vectorized version"""
			indices = self.np_random.choice(len(self.available_positions), size=self.num_drones, replace=False)
			positions = [self.available_positions[idx] for idx in indices]

			return positions

		if not self.is_vectorized:
			positions = _init_agent_positions()
		else:
			positions = [_init_agent_positions() for _ in range(self.num_vec_env)]

		return positions

	def compute_coverage_mask(
		self, positions: List[Tuple[int, int]] | List[List[Tuple[int, int]]]
	) -> np.ndarray:
		if not self.is_vectorized:
			return self.dataset._compute_coverage_mask(positions)
		else:
			masks = [self.dataset._compute_coverage_mask(pos) for pos in positions]
			return np.stack(masks, axis=0)

	def _get_pred_with_new_obs(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
		data_dict = dict()

		# get pad data from the predictor's scaler mean
		all_obs = self.observation_history + [observation]
		observed_time_steps = len(all_obs)
		if observed_time_steps >= self.in_time_steps:
			in_data = [all_obs[idx] for idx in range(-self.in_time_steps, 0)]
		else:
			in_data = all_obs

		source = torch.tensor(np.concatenate([obs['observed_traffic'] for obs in in_data], axis=0))
		source_steps = source.shape[-3] # works for vectorized (N T P C) shape and non-vectorized (T P C) shape
		padded_source = self.dataset.pad_backward_time(
			source=source, pad_len=self.predictor_input_steps - source_steps, zero_pad=True
		)
		# the predictor expects batched input, so in non-vectorized mode we need to add a batch dimension
		data_dict['source'] = padded_source if self.is_vectorized else padded_source.unsqueeze(0)

		with torch.no_grad():
			pred = self.predictor(data_dict)

		return {k: v for k, v in pred.items()}  # compress the batch dimension if it is 1

	def get_traffic_obs_pred(self, positions: List[Tuple[int, int]]) -> Dict[str, np.ndarray]:
		"""get the observed traffic at the given positions, and the predicted traffic based on the observation"""

		# clean data from the dataset, for 1 time step
		# shape (in_steps, num_locations, feature_dim)
		observed_traffic = self.data_sample['source'].detach().cpu().numpy()
		coverage_mask = self.compute_coverage_mask(
			positions
		)  # shape (num_locations,) or (num_vec_env, num_locations)

		# now we modify the nan values to be 0 to align with the observation space definition
		observed_traffic = np.nan_to_num(observed_traffic, nan=0.0)
		# zero-out unobserved locations (skip last feature for time encoding)
		if not self.is_vectorized:
			observed_traffic[..., coverage_mask == 0, :-1] = 0.0
		else:
			mask = np.asarray(coverage_mask)[:, None, :, None]
			observed_traffic[..., :-1] *= mask

		observation = {
			'observed_traffic': observed_traffic,
			'coverage_mask': coverage_mask,
		}

		pred_dict = self._get_pred_with_new_obs(observation)
		pred_traffic = pred_dict['pred'].detach().cpu().numpy()
		observation['batch_pred'] = pred_traffic.squeeze() if not self.is_vectorized else pred_traffic

		return observation

	def get_info(self) -> Dict | List[Dict]:
		if not self.is_vectorized:
			info = {
				'positions': list(self.positions),
				'step_index': self.step_index,
				'session_index': self.dataset.active_session,
				'gt': self.data_sample['target'].detach().cpu().numpy(),
			}
		else:
			info = []
			for env_idx, (pos, session_idx) in enumerate(zip(self.positions, self.dataset.active_session)):
				info.append({
					'positions': list(pos),
					'step_index': self.step_index,
					'session_index': session_idx,
					'gt': self.data_sample['target'][env_idx].detach().cpu().numpy(),
				})

		return info


	def update_history(self, positions: List[Tuple[int, int]], observation: Dict[str, np.ndarray]) -> None:
		self.positions = positions
		self.positions_history.append(tuple(self.positions))
		self.observation_history.append(observation)

		assert len(self.observation_history) == (self.step_index + 1), 'Observation history length mismatch.'

	def apply_actions(self, actions: np.array) -> List[Tuple[int, int]]:
		def _apply_single_env(
			actions: List[DroneAction], current_pos: List[Tuple[int, int]]
		) -> List[Tuple[int, int]]:
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
			return [DroneAction(a) for a in arr.tolist()]

		if not self.is_vectorized:
			new_positions = _apply_single_env(_to_act_space(actions), self.positions)
		else:
			new_positions = [
				_apply_single_env(_to_act_space(act), pos) for act, pos in zip(actions, self.positions)
			]

		return new_positions

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


class TrafficMonitorEnvVectorized(VecEnv):
	def __init__(self, dataset, predictor, num_drones: int, num_envs: int):

		assert num_envs > 1, 'num_envs should be greater than 1 for vectorized environment.'

		core_env = TrafficMonitorEnv(
			dataset=dataset,
			predictor=predictor,
			num_drones=num_drones,
			num_vec_env=num_envs,
			)
		self.core = core_env

		super().__init__(num_envs, core_env.observation_space, core_env.action_space)
		self._actions = None
		self.info_buffer: List[Dict] = [{} for _ in range(self.num_envs)]

	def reset(self):
		obs, self.reset_infos = self.core.reset()
		return obs

	def step_async(self, actions):
		self._actions = actions

	def step_wait(self) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[Dict]]:
		# TODO now it's a dummy implementation ... works with the interface but does nothing
		# The return type should be VecEnvStepReturn
		# VecEnvStepReturn = tuple[VecEnvObs, np.ndarray, np.ndarray, list[dict]]
		# where VecEnvObs = Union[np.ndarray, dict[str, np.ndarray], tuple[np.ndarray, ...]]

		if self._actions is None:
			raise RuntimeError('step_async() must be called before step_wait().')

		obs, rewards, terminated, truncated, self.info_buffer = self.core.step(self._actions)

		done_flag = bool(terminated or truncated)
		dones = np.full(self.num_envs, done_flag, dtype=bool)
		rewards = np.array(rewards, dtype=np.float32)

		if done_flag:
			for env_idx in range(self.num_envs):
				self.info_buffer[env_idx]['terminal_observation'] = {k: v[env_idx] for k, v in obs.items()}
			obs = self.reset()

		self._actions = None

		return obs, rewards, dones, deepcopy(self.info_buffer)

	def close(self):
		self.core.close()

	def get_attr(self, attr_name: str, indices=None) -> List[Any]:
		values = self._value_by_env(getattr(self.core, attr_name))
		return [values[i] for i in self._get_indices(indices)]

	def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
		# underlying env holds shared state, so we set once
		setattr(self.core, attr_name, value)

	def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> List[Any]:
		result = getattr(self.core, method_name)(*method_args, **method_kwargs)
		values = self._value_by_env(result)
		return [values[i] for i in self._get_indices(indices)]

	def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices=None) -> List[bool]:
		flag = isinstance(self.core, wrapper_class)
		return [flag for _ in self._get_indices(indices)]


if __name__ == "__main__":
	from skymonitor.agents import MonitoringAgent
	from skytraffic.utils.event_logger import setup_logger
	from stable_baselines3.common.env_checker import check_env

	logger = setup_logger(name="default", log_file="./scratch/drone_monitor_env.log", level=logging.INFO)

	num_drones = 10
	num_vec_env = None
	dataset = SimBarcaExplore(
		split="train",
		input_window=3,
		pred_window=30,
		step_size=3,
		num_unpadded_samples=20,
		allow_shorter_input=False,
		pad_input=False,
		norm_tid=False,
		vectorized=num_vec_env is not None,
	)
	predictor = TrafficPredictor(device='cuda') # looks like I don't really need this wrapper class ...

	env = TrafficMonitorEnv(
		dataset=dataset,
		predictor=predictor,
		num_drones=num_drones,
		num_vec_env=num_vec_env,
	)

	grid_info = {"grid_xy": dataset.grid_xy, "grid_id": dataset.grid_id, "grid_xy_to_id": dataset.grid_xy_to_id}
	agent = MonitoringAgent(num_drones=num_drones, grid=grid_info, policy_net=None)

	observation, info = env.reset()
	logger.info("Initial observation keys:{}".format(observation.keys()))
	logger.info("Initial coverage ratio: {}".format(observation["coverage_mask"].mean()))
	env.step(agent.select_action(observation))  # test step
	
	check_env(env, warn=True) # doesn't work for vectorized env