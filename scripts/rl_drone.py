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
from typing import Dict, Optional, List, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn

from simbarca_explore import SimBarcaExplore

logger = logging.getLogger("default")

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

    def __init__(self, input_steps, output_steps, data_channels, data_stats):
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.history_buffer = []
        self.prediction_network: nn.Module = None
        self._rng = torch.Generator()
    
    def predict(self, new_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        future = new_data.get("future_traffic")
        if future is None:
            raise ValueError("Sample must include 'future' tensor for prediction shape.")
        predictions = torch.rand(
            future.shape,
            dtype=future.dtype,
            device=future.device,
            generator=self._rng,
        )
        return {"future": predictions}
    
    def pack_history(self, new_data: List[Dict[str, torch.Tensor]]) -> None:
        """
        Pack the data history into the tensor input required by the neural network
        The history includes:
            observation masks for the traffic states. 
            clean traffic states with the traffic states of all road segments

        The running mask starts fully observed because the dataset pads the historical window with
        mean values. As new coverage arrives we roll the mask forward, filling the most recent
        timestep with the latest coverage booleans so unobserved locations are flagged as False.
        """
        pass 


class RewardCalculator:

    def __init__(self):
        self._rng = np.random.default_rng()
    
    def compute(self, data: Dict) -> float:
        """
        Compute reward based on predictor + newly collected data.
        Could be uncertainty reduction, entropy drop, etc.
        """
        return float(self._rng.random())
    
    def __call__(self, data):
        return self.compute(data)


class MonitoringAgent:
    def __init__(self, num_drones: int):
        self.policy_net: nn.Module = None
        self.num_drones = int(num_drones)
        self._rng = np.random.default_rng()
        self.positions: List[Tuple[int, int]] = None

        self.action_space = spaces.MultiDiscrete([len(DroneAction)] * self.num_drones)

    def select_action(self, obs: Dict) -> List[DroneAction]:
        """Choose next drone actions given observation for all drones."""
        return [DroneAction(x.item()) for x in self.action_space.sample()]

    
    def update(self, experiences: List[Tuple]) -> None:
        """Update policy from collected experience (RL training)."""
        return None



class DroneTrafficEnv(gym.Env):
    """ Design principles for the environment: 
        - **🎯 Agent Skill**: Collect traffic data in a grid map
        - **👀 Information**: Predicted traffic states, agent position, map states
        - **🎮 Actions**: Move up, down, left or right by 2 steps; Move diagonally by 1 step; Don't move
        - **🏆 Success**: the predictor has a smaller error compared to a naive flight plan
        - **⏰ End**: A simulation session ends.
    """
    metadata = {"render_modes": ["human"], "name": "drone_traffic_v0", "render_fps": 30}

    def __init__(
        self,
        dataset: SimBarcaExplore,
        agent: MonitoringAgent,
        predictor: TrafficPredictor,
        reward: RewardCalculator,
        
    ):
        super().__init__()
        self.dataset = dataset
        self.agent = agent
        self.predictor = predictor
        self.reward_fn = reward

        in_steps, num_locations, feature_dim = self.dataset.metadata['input_size']
        self.in_steps = int(in_steps)
        self.num_locations = int(num_locations)
        self.feature_dim = int(feature_dim)

        self.future_horizon = int(self.predictor.output_steps)
        self.max_steps = self.dataset.sample_per_session

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
            # DroneAction.UP_UP.value: (0, 2), # --- IGNORE ---
            # DroneAction.DOWN_DOWN.value: (0, -2), # --- IGNORE ---
            # DroneAction.LEFT_LEFT.value: (-2, 0), # --- IGNORE ---
            # DroneAction.RIGHT_RIGHT.value: (2, 0), # --- IGNORE ---
            # DroneAction.UP_LEFT.value: (-1, 1), # --- IGNORE ---
            # DroneAction.UP_RIGHT.value: (1, 1), # --- IGNORE ---
            # DroneAction.DOWN_LEFT.value: (-1, -1), # --- IGNORE ---
            # DroneAction.DOWN_RIGHT.value: (1, -1), # --- IGNORE ---
        }

        self.num_drones = agent.num_drones
        self.action_space = spaces.MultiDiscrete([len(DroneAction)] * self.num_drones)
        self.observation_space = spaces.Dict(
            {
                "observed_traffic": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(self.in_steps, self.num_locations, self.feature_dim),
                    dtype=np.float32,
                ),
                "coverage_mask": spaces.MultiBinary(self.num_locations),
                "step_index": spaces.Discrete(self.max_steps + 1),
            }
        )

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
        self.positions_history.append(self.positions)

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

        self.step_index, self.data_sample = next(self.data_iterator)

        self.positions = self.apply_actions(actions)

        observation = self.build_observation(self.positions)

        pred = self.predictor.predict(observation)

        reward_value = float(self.reward_fn(pred))

        self.positions_history.append(list(self.positions))

        terminated_flag = ( self.step_index >= (self.max_steps - 1) )
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
        self.agent.positions = positions
        return positions


    def _compute_coverage_mask(self, positions: List[Tuple[int, int]]) -> np.ndarray:
        """
        compute a boolean mask of shape (num_locations,) indicating which locations are covered
        at the drones at current `positions`
        """
        # simply add up the per-drone coverage masks. 
        # If a location is covered by at least one drone, it is considered covered.
        mask = sum([self.dataset.grid_id==self.dataset.grid_xy_to_id[(x,y)] for x, y in positions])

        return mask
    
    def build_observation(self, positions: List[Tuple[int, int]]) -> Dict[str, np.ndarray]:

        observation = {
            "observed_traffic": self.data_sample["past"],
            "future_traffic": self.data_sample["future"],
            "coverage_mask": self._compute_coverage_mask(positions),
            "step_index": self.step_index,
        }
        return observation
    
    def apply_actions(self, actions: List[DroneAction]) -> List[Tuple[int, int]]:
        action_list = actions.tolist() if isinstance(actions, np.ndarray) else list(actions)
        new_positions: List[Tuple[int, int]] = []
        occupied = set()
        for idx, action in enumerate(action_list):
            pos = self._get_new_position(action, self.positions[idx])
            if pos in occupied: # stay if the new position is already occupied
                pos = self.positions[idx]
            new_positions.append(pos)
            occupied.add(pos)
        self.agent.positions = new_positions
        return new_positions

    def _get_new_position(self, action: DroneAction, old_pos: Tuple[int, int]) -> Tuple[int, int]:

        direction = self._action_to_direction.get(action, (0,0))
        new_pos = (old_pos[0] + direction[0], old_pos[1] + direction[1])
        if new_pos not in self.available_positions:
            new_pos = old_pos
        return new_pos


    def render(self):
        """Visualize drone paths with matplotlib in terminal mode."""

        import matplotlib.pyplot as plt
        from matplotlib import animation

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

    dataset = SimBarcaExplore(split="train", input_window=3, pred_window=30, step_size=3, allow_shorter_input=False, pad_input=False)
    dataset.active_session = 0
    predictor = TrafficPredictor(
        input_steps=dataset.metadata["input_size"][0],
        output_steps=dataset.metadata["output_size"][0],
        data_channels=dataset.metadata["data_channels"],
        data_stats=dataset.metadata["data_stats"],
    )
    reward_calculator = RewardCalculator()
    agent = MonitoringAgent(num_drones=10)

    env = DroneTrafficEnv(dataset=dataset, predictor=predictor, agent=agent, reward=reward_calculator)

    observation, info = env.reset()
    print("Initial observation keys:", observation.keys())
    print("Initial coverage ratio:", observation["coverage_mask"].mean())

    done = False
    episode_reward = 0.0
    step_count = 0
    while not done and step_count < 5:
        actions = agent.select_action(observation)
        observation, reward, done, _, step_info = env.step(actions)
        episode_reward += reward
        step_count += 1
        print(f"Step {step_count}: reward={reward:.3f}, done={done}, coverage={observation['coverage_mask'].mean():.3f}")

    from matplotlib import animation as mpl_animation
    anim = env.render()
    writer = mpl_animation.PillowWriter(fps=1)
    anim.save("drone_paths.gif", writer=writer, dpi=120)
    print("Saved animation to drone_paths.gif at 5 fps")

    env.close()
