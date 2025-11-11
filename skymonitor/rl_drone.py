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
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib import animation

import torch
import torch.nn as nn

from skytraffic.utils.event_logger import setup_logger
from skymonitor.simbarca_explore import SimBarcaExplore
from skymonitor.patch_lgc import PatchedMVLSTMGCNConv

logger = setup_logger(name="default", log_file="./scratch/rl_drone.log", level=logging.INFO)

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


class TrafficPredictor:
    """ Basically a wrapper class for the neural network, the RL environment gives observation per-time-step (3 mins),
        but the predictor will need a context window (e.g., 30 mins) to make predictions.
    """

    def __init__(self, device: Optional[torch.device] = "cpu"):
        self.device = torch.device(device)
        self.net: nn.Module = PatchedMVLSTMGCNConv(
            use_global=True,
            feature_dim=3,
            d_model=64,
            temp_patching=3,
            global_downsample_factor=1,
            layernorm=True,
            adjacency_hop=1,
            dropout=0.1,
            loss_ignore_value=float("nan"),
            norm_label_for_loss=True,
            input_steps=360,
            pred_steps=10,
            num_nodes=1570,
            pred_feat=2,
            data_null_value=0.0,
        )
        state_dict = torch.load("./scratch/patch_lgc_simbarca_explore/model_final.pth")
        self.net.load_state_dict(state_dict['model'])
        self.net.eval()
        self.net.to(self.device)
        self.in_steps = self.net.input_steps
        self.output_steps = self.net.pred_steps

    def predict(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:

        predictions = self.net(data_dict)

        return predictions

    def __call__(self, new_data: Dict):
        return self.predict(new_data)


class CentralizedMonitoringAgent:

    def __init__(self, num_drones: int, grid: Dict = None):
        self.policy_net: nn.Module = None
        self.num_drones = int(num_drones)
        self.positions: List[Tuple[int, int]] = None
        self.action_space: gym.Space = None
        # the grid_id of all road segments, shape (num_locations,)
        self.grid_id = grid.get("grid_id", None)
        # the grid (x, y) coordinates of all road segments, shape (num_locations, 2)
        self.grid_xy = grid.get("grid_xy", None)
        # Dict[Tuple[int, int], int], the mapping from (x, y) to grid_id
        # the keys are non-empty grid (x,y) coordinates, the values are the ids of non-empty grids
        self.grid_xy_to_id = grid.get("grid_xy_to_id", None)
        self.id_of_non_empty_grids = list(self.grid_xy_to_id.values())
        self.xy_of_non_empty_grids = list(self.grid_xy_to_id.keys())

    def bind_action_space(self, action_space: spaces.Space) -> None:
        """ The action space of the agent should match that of the environment. """
        self.action_space = action_space

    def select_action(self, obs: Dict, info: Dict) -> List[DroneAction]:
        """ Choose next drone actions given observation for all drones.
            Args:
                obs: Dict, the observation from the environment, from function `TrafficMonitorEnv.build_observation`
                info: Dict, the info from `TrafficMonitorEnv.step`

            return: List[DroneAction], the actions for all drones
        """
        # shape (in_steps, num_locations, feature_dim), the observed traffic for the recent time step (3 mins)
        # drone data has high frequency (5s), so we have in_steps = 36 for 3 mins
        # in feature_dim, we have (flow, density, time-in-day), time-in-day is normalized to [0,1], e.g., 1/3 means 8am
        current_traffic = obs['observed_traffic']
        # shape (num_locations,), a True means the location is covered by drones
        coverage_mask = obs['coverage_mask']
        # predicted traffic for next 30 minutes, with 3 min time step 
        # shape (batch_size, out_steps, num_locations, feature_dim)
        predicted_traffic = info['batch_pred'].squeeze()

        predicted_traffic_at_grid_x9y7 = predicted_traffic[:, self.grid_id==self.grid_xy_to_id[(9,7)], :]
        flow_at_x9y7 = predicted_traffic_at_grid_x9y7[..., 0] # shape (out_steps, road_segments_in_grid_x9y7)
        density_at_x9y7 = predicted_traffic_at_grid_x9y7[..., 1]  # shape (out_steps, road_segments_in_grid_x9y7)

        
        predicted_traffic_at_grid_id = predicted_traffic[:, self.grid_id==self.id_of_non_empty_grids[77], :]
        flow_at_grid_id_42 = predicted_traffic_at_grid_id[..., 0]  # shape (out_steps, road_segments_in_grid_id_42)
        density_at_grid_id_42 = predicted_traffic_at_grid_id[..., 1]  # shape (out_steps, road_segments_in_grid_id_42)

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
        num_drones: int,
    ):
        super().__init__()
        self.dataset = dataset
        self.predictor = predictor
        self.baseline_agent = None

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
                    shape=(self.in_data_steps, self.num_locations, self.feature_dim),
                    dtype=np.float32,
                ),
                "coverage_mask": spaces.MultiBinary(n=self.num_locations),
                # "empty_mask": spaces.MultiBinary(n=[self.in_data_steps, self.num_locations]),
            }
        )
        self.step_index: int = 0
        self.data_sample: Dict[str, torch.Tensor] = None
        self.positions: List[Tuple[int, int]] = []
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
        if isinstance(options, dict):
            self.dataset.active_session = options.get("active_session", self._rng.integers(0, self.dataset.num_sessions))
        else:
            self.dataset.active_session = self._rng.integers(0, self.dataset.num_sessions)

        self.data_iterator = self.dataset.iterate_active_session()
        self.step_index, self.data_sample = next(self.data_iterator)
        assert self.step_index == 0, "Step index should start from 0 after reset."

        self.positions = self.init_agent_positions()

        self.positions_history.append(list(self.positions))

        observation = self.build_observation(self.positions)

        self.observation_history.append(observation)

        pred = self.predictor(self.pack_predictor_context())

        info = {
            "positions": list(self.positions),
            "step_index": 0,
            "session_index": self.dataset.active_session,
            "batch_pred": pred['pred'].detach().cpu(),
            "batch_gt": self.data_sample['target'].detach().cpu().unsqueeze(0),
        }

        return observation, info

    def step(self, actions: List[DroneAction]) -> Tuple[Dict, float, bool, bool, Dict]:
        try:
            self.step_index, self.data_sample = next(self.data_iterator)
        except StopIteration: # indicate the end of the episode
            return dict(), 0.0, True, False, {}

        self.positions = self.update_positions(actions)

        observation = self.build_observation(self.positions)

        self.positions_history.append(list(self.positions))

        self.observation_history.append(observation)

        pred = self.predictor(self.pack_predictor_context())

        reward_value = float(self.calculate_reward(observation, pred, self.data_sample))

        terminated_flag = ( self.step_index >= (self.max_steps_per_session - 1) )
        truncated_flag = False

        info = {
            "positions": list(self.positions),
            "step_index": self.step_index,
            "session_index": self.dataset.active_session,
            "batch_pred": pred['pred'].detach().cpu(),
            "batch_gt": self.data_sample['target'].detach().cpu().unsqueeze(0),
        }

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
        observation: Dict[str, np.ndarray],
        prediction: Dict[str, torch.Tensor],
        ground_truth: Dict[str, torch.Tensor],
    ) -> float:
        """ The agent will be rewarded primariy based on improvements of prediction quality:
                ** The predictor achieves a smaller error compared to a baseline agent (e.g., random flight plan) **
            And in addition, we add rewards to encourage:
                1. larger coverage
                2. ??? 
        """
        return 0.0

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
            # "empty_mask": empty_mask,
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

    def pack_predictor_context(self) -> Dict[str, torch.Tensor]:
        """ Pad the input data to proper size if needed.
            Input data shape: (in_steps, num_locations, in_channels)
        """
        data_dict = dict()

        # get pad data from the predictor's scaler mean
        observed_time_steps = len(self.observation_history)
        if observed_time_steps >= self.in_time_steps:
            in_data = [self.observation_history[idx] for idx in range(-self.in_time_steps, 0)]
        else:
            in_data = self.observation_history

        source = torch.tensor(np.concat([obs["observed_traffic"] for obs in in_data], axis=0))
        padded_source = self.dataset.pad_backward_time(source=source, pad_len=self.predictor_input_steps-source.shape[0], zero_pad=True)
        data_dict['source'] = padded_source.unsqueeze(0)  # add batch dimension for pytorch model

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

def eval_monitoring_agent(env: TrafficMonitorEnv, agent: CentralizedMonitoringAgent):

    all_pred, all_gt = [], []
    for active_session in range(env.dataset.num_sessions):
        print("=== Running agents on session {} ===".format(active_session))
        observation, info = env.reset(options={"active_session": active_session})
        done = False
        episode_reward = 0.0
        step_count = 0
        episode_pred, episode_gt = [], []
        while not done:
            actions = agent.select_action(observation, info)
            observation, reward, done, _, info = env.step(actions)
            step_count += 1
            episode_reward += reward
            episode_pred.append(info['batch_pred'])
            episode_gt.append(info['batch_gt'])
        
        all_pred.append(torch.cat(episode_pred, dim=0))  # concat on batch dimension
        all_gt.append(torch.cat(episode_gt, dim=0))
    
    all_pred = torch.cat(all_pred, dim=0)
    all_gt = torch.cat(all_gt, dim=0)

    return all_pred, all_gt

if __name__ == "__main__":
    dataset = SimBarcaExplore(
        split="train",
        input_window=3,
        pred_window=30,
        step_size=3,
        num_unpadded_samples=20,
        allow_shorter_input=False,
        pad_input=False,
        norm_tid=False,
    )
    predictor = TrafficPredictor(device='cuda') # looks like I don't really need this wrapper class ...

    num_drones = 10
    env = TrafficMonitorEnv(
        dataset=dataset,
        predictor=predictor,
        num_drones=num_drones,
    )

    from stable_baselines3.common.env_checker import check_env
    check_env(env, warn=True)

    grid_info = {"grid_xy": dataset.grid_xy, "grid_id": dataset.grid_id, "grid_xy_to_id": dataset.grid_xy_to_id}
    agent = CentralizedMonitoringAgent(num_drones=num_drones, grid=grid_info)
    agent.bind_action_space(env.action_space)

    observation, info = env.reset()
    logger.info("Initial observation keys:{}".format(observation.keys()))
    logger.info("Initial coverage ratio: {}".format(observation["coverage_mask"].mean()))

    with torch.no_grad():
        all_pred, all_gt = eval_monitoring_agent(env, agent)

    from skymonitor.simbarca_explore_evaluation import SimBarcaExploreEvaluator
    evaluator = SimBarcaExploreEvaluator(
        save_dir="scratch/rl_drone_evaluation",
        visualize=True,
        ignore_value=0.0,
    )
    eval_results = evaluator.calculate_error_metrics(pred=all_pred, label=all_gt, data_channels=dataset.data_channels['target'])
    logger.info("Drone Monitoring Evaluation Results: {}".format(eval_results))

    env.close()
