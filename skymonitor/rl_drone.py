""" We consider flying a fleet of drones to collect traffic data (flow, density, speed) over a large urban area.
    The goal is to smartly plan the drone flights so that the collected data can allow a traffic predictor to achieve the best performance.
    We plan to develop an RL-based solution to this problem, which involves:
        1. A dataset `SimBarcaExplore` that provides traffic data of an urban area, where the space is divided to grids.
        2. A traffic predictor that takes in past observations and predict the future traffic states (flow, density, speed)
        3. A reward calculator based on the evaluation of traffic predictions (how good is the predictor doing)
        4. A set of monitoring agents (drones) to query data from the dataset
        5. An environment to orchestrate the four components above
"""
import logging
from enum import Enum
from typing import Dict, List, Tuple, Optional
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib import animation

import torch
import torch.nn as nn

from skytraffic.utils.event_logger import setup_logger
from skymonitor.simbarca_explore import SimBarcaExplore

logger = setup_logger(name="default", log_file="./project/rl_drone.log", level=logging.INFO)

class DroneAction(Enum):
    """ A drone can stay, move up/down/left/right by at most 2 steps, or move diagonally by 1 step.
        E.g., a drone can fly at 20 m/s, and a grid cell is 220 m, in a time step of 3 mins, 
        a drone can can spend ~20s to move and the rest of the time to hover and collect data.
    """
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    # # these may make the action space too large, start with simpler actions first
    # UP_UP = 5
    # DOWN_DOWN = 6
    # LEFT_LEFT = 7
    # RIGHT_RIGHT = 8
    # UP_LEFT = 9
    # UP_RIGHT = 10
    # DOWN_LEFT = 11
    # DOWN_RIGHT = 12


class TrafficPredictor:
    """ Basically a wrapper class for the neural network, the RL environment gives observation per-time-step (3 mins),
        but the predictor will need a context window (e.g., 30 mins) to make predictions.
        This class aims to handle the padding of input data.
    """

    def __init__(self, in_steps: int, in_channels: int, out_steps: int, out_channels: int):
        self.in_steps = in_steps
        self.in_channels = in_channels
        self.output_steps = out_steps
        self.out_channels = out_channels
        self.coverage_mask = None
        self.prediction_network: nn.Module = None
    
    def predict(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:

        predictions = {}

        return predictions

    def __call__(self, new_data: Dict):
        return self.predict(new_data)
    

class RewardCalculator:

    def __init__(self):
        # the agent is rewarded for outperforming a baseline policy
        self.baseline_policy: nn.Module = None
        self._rng = np.random.default_rng()
    
    def compute(self, observation:Dict, pred: Dict, label:torch.Tensor) -> float:
        """
        Compute reward based on predictor + newly collected data.
        Could be uncertainty reduction, entropy drop, etc.
        """
        return float(self._rng.random())
    
    def __call__(self, observation:Dict, pred: Dict, label:torch.Tensor) -> float:
        return self.compute(observation, pred, label)


class CentralizedMonitoringAgent:
    def __init__(self, num_drones: int):
        self.policy_net: nn.Module = None
        self.num_drones = int(num_drones)
        self._rng = np.random.default_rng()
        self.positions: List[Tuple[int, int]] = None
        self.action_space: gym.Space = None

    def bind_action_space(self, action_space: spaces.Space) -> None:
        """Call once to provide the environment's action space."""
        self.action_space = action_space

    def select_action(self, obs: Dict) -> List[DroneAction]:
        """Choose next drone actions given observation for all drones."""
        sampled = self.action_space.sample()
        return [DroneAction(int(x)) for x in sampled]

    def update(self, experiences: List[Tuple]) -> None:
        """Update policy from collected experience (RL training)."""
        return None



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
        reward: RewardCalculator,
        num_drones: int,
    ):
        super().__init__()
        self.dataset = dataset
        self.predictor = predictor
        self.reward_fn = reward

        in_steps, num_locations, feature_dim = self.dataset.metadata['input_size']
        self.in_steps = int(in_steps)
        self.num_locations = int(num_locations)
        self.feature_dim = int(feature_dim)

        self.future_horizon = int(self.predictor.output_steps)
        self.max_steps_per_session = self.dataset.sample_per_session

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

        self.num_drones = int(num_drones)
        self.action_space = spaces.MultiDiscrete([len(DroneAction)] * self.num_drones)
        self.observation_space = spaces.Dict(
            {
                "observed_traffic": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(self.in_steps, self.num_locations, self.feature_dim),
                    dtype=np.float32,
                ),
                "coverage_mask": spaces.MultiBinary(n=self.num_locations),
                "empty_mask": spaces.MultiBinary(n=[self.in_steps, self.num_locations]),
            }
        )
        self.step_index: int = 0
        self.data_sample: Dict[str, torch.Tensor] = None
        self.positions: List[Tuple[int, int]] = None
        self.positions_history = deque(maxlen=self.max_steps_per_session)
        self.observation_history: List[Dict[str, torch.Tensor]] = deque(maxlen=self.max_steps_per_session)

        self._last_animation = None
        self._rng: np.random.Generator = np.random.default_rng()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Resets environment and predictor.
           Returns observations and infos for all agents.
        """
        self.clear_state()

        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        if self.dataset.active_session is None:
            self.dataset.active_session = 0
        
        self.data_iterator = self.dataset.iterate_active_session()
        self.step_index, self.data_sample = next(self.data_iterator)
        assert self.step_index == 0, "Step index should start from 0 after reset."

        self.positions = self.init_agent_positions()
        self.positions_history.append(list(self.positions))

        observation = self.build_observation(self.positions)
        info = {
            "positions": list(self.positions),
            "step_index": 0,
            "session_index": self.dataset.active_session,
        }
        return observation, info

    def step(self, actions: List[DroneAction]) -> Tuple[Dict, float, bool, bool, Dict]:
        """ 
        """

        try:
            self.step_index, self.data_sample = next(self.data_iterator)
        except StopIteration: # indicate the end of the episode
            return dict(), 0.0, True, False, {}

        self.positions = self.update_positions(actions)

        observation = self.build_observation(self.positions)

        pred = self.predictor(self.pack_predictor_context(observation))

        reward_value = float(self.reward_fn(observation, pred, self.data_sample))

        self.positions_history.append(list(self.positions))
        self.observation_history.append(observation)

        terminated_flag = ( self.step_index >= (self.max_steps_per_session - 1) )
        truncated_flag = False

        info = {
            "positions": list(self.positions),
            "step_index": self.step_index,
            "session_index": self.dataset.active_session,
        }

        return observation, reward_value, terminated_flag, truncated_flag, info

    def close(self):
        self.clear_state()

    def clear_state(self) -> None:
        self.step_index = 0
        self.data_sample = None
        self.positions = []
        self.positions_history = []
        self._last_animation = None
        self._rng: np.random.Generator = None

    def init_agent_positions(self):
        """Initialize drone positions at start of episode."""
        indices = self._rng.choice(len(self.available_positions), size=self.num_drones, replace=False)
        positions = [self.available_positions[idx] for idx in indices]
        return positions

    def compute_coverage_mask(self, positions: List[Tuple[int, int]]) -> np.ndarray:
        return self.dataset._compute_coverage_mask(positions)
    
    def build_observation(self, positions: List[Tuple[int, int]]) -> Dict[str, np.ndarray]:

        # clean data from the dataset, for 1 time step 
        # shape (in_steps, num_locations, feature_dim)
        observed_traffic = self.data_sample["source"].cpu().numpy() 
        coverage_mask = self.compute_coverage_mask(positions) # shape (num_locations,)
        empty_mask = np.isnan(observed_traffic).any(axis=-1).astype(np.int8)
        # one can check the nans only exists in the speed measurements ... 
        # np.allclose(np.isnan(observed_traffic[:,:,2]), empty_mask)

        # now we modify the nan values to be 0 to align with the observation space definition
        observed_traffic = np.nan_to_num(observed_traffic, nan=0.0)
        # set the unobserved locations to be 0.0 as well, but don't touch the last feature dimension for time-in-day
        observed_traffic[:, coverage_mask==0, :-1] = 0.0

        observation = {
            "observed_traffic": observed_traffic,
            "coverage_mask": coverage_mask,
            "empty_mask": empty_mask,
        }

        return observation
    
    def update_positions(self, actions: List[DroneAction]) -> List[Tuple[int, int]]:
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

        direction = self._action_to_direction.get(action, (0,0))
        new_pos = (old_pos[0] + direction[0], old_pos[1] + direction[1])
        if new_pos not in self.available_positions:
            new_pos = old_pos
        return new_pos

    def pack_predictor_context(self, observation: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """ Pad the input data to proper size if needed.
            Input data shape: (in_steps, num_locations, in_channels)
        """
        data_dict = {} 

        # get pad data from the predictor's scaler mean

        # get pad time encoding by linearly counting back from the first time step in the input data
        # count_back_time = (
        #     torch.arange(0, pad_len * 5, step=5) - pad_len * 5
        # )  #  -pad_len*5, ... -10 ,-5
        # count_back_time = count_back_time / (24 * 60 * 60.0)  # seconds in a day
        # # count back from the first time step in the input data
        # count_back_time = time_in_day_5s[0, 0] + repeat(count_back_time, "t -> t n", n=source.shape[1])

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


if __name__ == "__main__":
    dataset = SimBarcaExplore(
        split="train",
        input_window=30,
        pred_window=30,
        step_size=3,
        num_unpadded_samples=20,
        allow_shorter_input=True,
        pad_input=False,
    )
    dataset.active_session = 0
    predictor = TrafficPredictor(
        in_steps=dataset.metadata['input_size'][0],
        in_channels=dataset.metadata['input_size'][2],
        out_steps=dataset.metadata['output_size'][0],
        out_channels=dataset.metadata['output_size'][2],
    )
    reward_calculator = RewardCalculator()
    num_drones = 10
    env = TrafficMonitorEnv(
        dataset=dataset,
        predictor=predictor,
        reward=reward_calculator,
        num_drones=num_drones,
    )
    from stable_baselines3.common.env_checker import check_env
    check_env(env, warn=True)

    agent = CentralizedMonitoringAgent(num_drones=num_drones)
    agent.bind_action_space(env.action_space)

    observation, info = env.reset()
    logger.info("Initial observation keys:{}".format(observation.keys()))
    logger.info("Initial coverage ratio: {}".format(observation["coverage_mask"].mean()))

    done = False
    episode_reward = 0.0
    step_count = 0
    while not done:
        actions = agent.select_action(observation)
        observation, reward, done, _, step_info = env.step(actions)
        episode_reward += reward
        step_count += 1
        logger.info(
            "Step {}: reward={:.3f}, done={}, coverage={:.3f}".format(
                step_count, reward, done, observation["coverage_mask"].mean()
            )
        )

    anim = env.render_episode()
    writer = animation.PillowWriter(fps=5)
    anim.save("drone_paths.gif", writer=writer, dpi=120)
    logger.info("Saved animation to drone_paths.gif")

    env.close()
