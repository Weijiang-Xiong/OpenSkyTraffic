""" Gymnasium environment for SkyMonitor.
    see the `Gymnasium environment creation tutorial` at the link:
    https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib import animation

import torch

from skymonitor.simbarca_explore import SimBarcaExplore
from skymonitor.agents import RandomAgent, DroneAction
from skymonitor.traffic_predictor import TrafficPredictor

class TrafficMonitorEnv(gym.Env):
    """ Design principles for the environment: 
        - **🎯 Agent Skill**: Collect traffic data in a grid map
        - **👀 Information**: Predicted traffic states, agent position, map structure
        - **🎮 Actions**: Move up, down, left or right by 1 or 2 steps; Move diagonally by 1 step; Don't move
        - **🏆 Success**: the predictor has a smaller error compared to a naive flight plan
        - **⏰ End**: A simulation session ends.
    """
    metadata = {"render_modes": ["human"], "name": "drone_traffic_v0", "render_fps": 30}

    def __init__(
        self,
        dataset: SimBarcaExplore,
        predictor: TrafficPredictor,
        action_space: spaces.MultiDiscrete,
    ):
        super().__init__()
        # components of the environment
        self.dataset = dataset
        self.predictor = predictor
        self.baseline_agent = RandomAgent(action_space=action_space)

        # data shape info
        in_data_steps, num_locations, feature_dim = self.dataset.metadata['input_size']
        # we move our drones every `time_step` (3 min), and the input drone data are given every data_step (5s)
        self.in_time_steps = int(self.dataset.input_window // self.dataset.step_size)
        self.in_data_steps = int(in_data_steps)
        self.data_pt_per_time_step = int(self.in_data_steps // self.in_time_steps)
        self.num_locations = int(num_locations)
        self.feature_dim = int(feature_dim)
        self.predictor_input_steps = int(self.predictor.in_steps)
        self.future_horizon = int(self.predictor.output_steps)
        self.max_steps_per_session = self.dataset.sample_per_session

        # grid structure info
        self.grid_ids:np.ndarray = self.dataset.grid_id
        self.grid_xy_to_id: Dict[Tuple[int, int], int] = self.dataset.grid_xy_to_id
        # empty grid cells are not valid positions
        self.available_positions = list(self.grid_xy_to_id.keys())
        self._action_to_direction = {
            DroneAction.STAY: (0, 0),
            DroneAction.UP: (0, 1),
            DroneAction.DOWN: (0, -1),
            DroneAction.LEFT: (-1, 0),
            DroneAction.RIGHT: (1, 0),
        }
        self.num_drones = int(action_space.nvec.shape[0])

        # action and observation spaces, required by gym.Env
        self.action_space = action_space
        self.observation_space = spaces.Dict(
            {
                "observed_traffic": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=self.dataset.input_size,
                    dtype=np.float32,
                ),
                "coverage_mask": spaces.MultiBinary(n=self.num_locations),
                "batch_pred": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=self.dataset.output_size,
                    dtype=np.float32,
                ),
                "batch_gt": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=self.dataset.output_size,
                    dtype=np.float32,
                ),
            }
        )

        # running time state variables
        self.step_index: int = 0
        self.data_sample: Dict[str, torch.Tensor] = None
        self.positions: List[Tuple[int, int]] = []
        self.positions_history = list()
        self.observation_history: List[Dict[str, torch.Tensor]] = list()
        self._last_animation = None
        self._rng: np.random.Generator = np.random.default_rng()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Resets environment and predictor.
           Returns observations and infos for all agents.
        """
        self.clear_state()
        super().reset(seed=seed)

        self._rng = np.random.default_rng(seed)
        if isinstance(options, dict):
            self.dataset.active_session = options.get("active_session", self._rng.integers(0, self.dataset.num_sessions))
        else:
            self.dataset.active_session = self._rng.integers(0, self.dataset.num_sessions)

        self.data_iterator = self.dataset.iterate_active_session()
        self.step_index, self.data_sample = next(self.data_iterator)
        assert self.step_index == 0, "Step index should start from 0 after reset."

        positions = self.init_agent_positions()
        observation = self.build_observation(positions)
        self.b_actions = self.baseline_agent.select_action(observation)

        self.update_history(positions, observation)
        info = self.build_info()

        return observation, info

    def step(self, actions: List[DroneAction]) -> Tuple[Dict, float, bool, bool, Dict]:
        try:
            self.step_index, self.data_sample = next(self.data_iterator)
        except StopIteration: # indicate the end of the episode
            return dict(), 0.0, True, False, {}

        positions = self.get_new_positions(actions)
        observation = self.build_observation(positions)
        self.b_actions = self.baseline_agent.select_action(observation)

        reward_value = float(self.calculate_reward(observation['batch_pred'], self.data_sample))

        terminated_flag = ( self.step_index >= (self.max_steps_per_session - 1) )
        truncated_flag = False

        self.update_history(positions, observation)
        info = self.build_info()

        return observation, reward_value, terminated_flag, truncated_flag, info

    def close(self):
        self.clear_state()

    def clear_state(self) -> None:
        self.step_index = 0
        self.data_sample = None
        self._last_animation = None
        self._rng: np.random.Generator = None
        self.positions.clear()
        self.positions_history.clear()
        self.observation_history.clear()

    def calculate_reward(
        self,
        prediction: Dict[str, torch.Tensor],
        ground_truth: Dict[str, torch.Tensor],
    ) -> float:
        """ The agent will be rewarded primariy based on improvements of prediction quality:
                ** The predictor achieves a smaller error compared to a baseline agent (e.g., random flight plan) **
            And in addition, we add rewards to encourage:
                1. larger coverage
                2. ??? 
        """

        b_positions = self.get_new_positions(self.b_actions)
        b_observation = self.build_observation(b_positions)

        # prediction with the baseline agent's observation
        # the data collected by the smart agent should be better than the baseline agent
        b_pred = self._get_pred_with_new_obs(b_observation)

        return 0.0

    def init_agent_positions(self):
        """Initialize drone positions at start of episode."""
        indices = self._rng.choice(len(self.available_positions), size=self.num_drones, replace=False)
        positions = [self.available_positions[idx] for idx in indices]
        return positions

    def compute_coverage_mask(self, positions: List[Tuple[int, int]]) -> np.ndarray:
        return self.dataset._compute_coverage_mask(positions)
    
    def _get_pred_with_new_obs(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            pred = self.predictor(self.pack_predictor_context(observation))
        return pred

    def build_observation(self, positions: List[Tuple[int, int]]) -> Dict[str, np.ndarray]:

        # clean data from the dataset, for 1 time step 
        # shape (in_steps, num_locations, feature_dim)
        observed_traffic = self.data_sample["source"].detach().cpu().numpy() 
        coverage_mask = self.compute_coverage_mask(positions) # shape (num_locations,)
        # only useful for speed channel, but since we exclude it from dataset, we don't need it for now
        # empty_mask = np.isnan(observed_traffic[..., self.dataset.data_channels['source'].index('density')]).any(axis=-1).astype(np.int8)
        # one can check the nans only exists in the speed measurements ... 
        # np.allclose(np.isnan(observed_traffic[:,:,2]), empty_mask)

        # now we modify the nan values to be 0 to align with the observation space definition
        observed_traffic = np.nan_to_num(observed_traffic, nan=0.0)
        # set the unobserved locations to be 0.0 as well, but don't touch the last feature dimension for time-in-day
        observed_traffic[:, coverage_mask==0, :-1] = 0.0

        observation = {
            "observed_traffic": observed_traffic,
            "coverage_mask": coverage_mask,
        }

        pred = self._get_pred_with_new_obs(observation)
        observation['batch_pred'] = pred['pred'].detach().cpu().numpy()
        observation["batch_gt"] = self.data_sample['target'].detach().cpu().numpy()

        return observation
    
    def build_info(self) -> Dict:

        info = {
            "positions": list(self.positions),
            "step_index": self.step_index,
            "session_index": self.dataset.active_session,
        }

        return info
    
    def update_history(self, positions: List[Tuple[int, int]], observation: Dict[str, np.ndarray]) -> None:
        
        self.positions = positions
        self.positions_history.append(list(self.positions))
        self.observation_history.append(observation)

        assert len(self.observation_history) == (self.step_index + 1), "Observation history length mismatch."

    def get_new_positions(self, actions: List[DroneAction]) -> List[Tuple[int, int]]:
        new_positions: List[Tuple[int, int]] = []
        occupied = set()
        for idx, action in enumerate(actions):
            pos = self._get_new_position(DroneAction(action), self.positions[idx])
            if pos in occupied: # stay if the new position is already occupied
                pos = self.positions[idx]
            new_positions.append(pos)
            occupied.add(pos)
        return new_positions

    def _get_new_position(self, action: DroneAction, old_pos: Tuple[int, int]) -> Tuple[int, int]:
        """ works for 1 single position update """
        direction = self._action_to_direction.get(action, (0,0))
        new_pos = (old_pos[0] + direction[0], old_pos[1] + direction[1])
        if new_pos not in self.available_positions:
            new_pos = old_pos
        return new_pos

    def pack_predictor_context(self, new_obs: Dict) -> Dict[str, torch.Tensor]:
        """ Pad the input data to proper size if needed.
            Input data shape: (in_steps, num_locations, in_channels)
        """
        data_dict = dict()

        # get pad data from the predictor's scaler mean
        all_obs = self.observation_history + [new_obs]
        observed_time_steps = len(all_obs)
        if observed_time_steps >= self.in_time_steps:
            in_data = [all_obs[idx] for idx in range(-self.in_time_steps, 0)]
        else:
            in_data = all_obs

        source = torch.tensor(np.concatenate([obs["observed_traffic"] for obs in in_data], axis=0))
        padded_source = self.dataset.pad_backward_time(source=source, pad_len=self.predictor_input_steps-source.shape[0], zero_pad=True)
        # the padded source will be either shape (N, T, P, C) or (T, P, C), where T is predictor_input_steps
        if padded_source.shape[0] == self.predictor_input_steps:
            data_dict['source'] = padded_source.unsqueeze(0)  

        return data_dict

    def render(self):
        pass


    def render_episode(self):
        """Visualize drone paths with matplotlib in terminal mode."""

        if not self.positions_history:
            return None

        positions_frames = [np.asarray(frame) for frame in self.positions_history]
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
            ax.axvline(x - 0.5, color="gray", linewidth=0.5, alpha=0.4)
        for y in range(grid_limits[1] + 1):
            ax.axhline(y - 0.5, color="gray", linewidth=0.5, alpha=0.4)
        ax.set_title("Drone Coverage Summary")
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")

        # initialize empty scatter plots, trails and annotations. 
        scatter = ax.scatter([], [], c="tab:red", s=80, label="Drones")
        trails = [ax.plot([], [], linestyle="-", marker="o", alpha=0.5)[0] for _ in range(self.num_drones)]
        for (x, y), label in zip(grid_xy, self.dataset.grid_id):
            ax.text(
                x,
                y,
                str(label),
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
        annotation = ax.text(
            0.02,
            0.95,
            "",
            transform=ax.transAxes,
            fontsize=10,
            ha="left",
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
        )

        def init():
            annotation.set_text("")
            for line in trails:
                line.set_data([], [])
            return [scatter, annotation, *trails]

        def update(frame_idx):
            scatter.set_offsets(positions_frames[frame_idx])
            annotation.set_text(f"Step {frame_idx} / {steps - 1}")
            for drone_idx, line in enumerate(trails):
                history = np.asarray([
                    frame[drone_idx] for frame in positions_frames[: frame_idx + 1]
                ])
                line.set_data(history[:, 0], history[:, 1])
            return [scatter, annotation, *trails]

        base_interval = 1000 / max(1, self.metadata.get("render_fps", 30))
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
