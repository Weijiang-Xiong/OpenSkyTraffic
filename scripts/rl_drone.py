""" We consider flying a fleet of drones to collect traffic data (flow, density, speed) over a large urban area.
    The goal is to smartly plan the drone flights so that the collected data can allow a traffic predictor to achieve the best performance.
    We plan to develop an RL-based solution to this problem, which involves:
        1. A dataset `SimBarcaExplore` that provides traffic data of an urban area, where the space is divided to grids.
        2. A traffic predictor that takes in past observations and predict the future traffic states (flow, density, speed)
        3. A reward calculator based on the evaluation of traffic predictions (how good is the predictor doing)
        4. A set of monitoring agents (drones) to query data from the dataset
        5. An environment to orchestrate the four components above
"""
import os
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
from einops import repeat

from skytraffic.data.datasets.simbarca_base import SimBarcaForecast

logger = logging.getLogger("default")

class DroneTrafficEnv(gym.Env):
    
    metadata = {"render_modes": ["human"], "name": "drone_traffic_v0", "render_fps": 30}

    def __init__(self, dataset, predictor, agent, reward):
        super().__init__()


    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        """Resets environment and predictor.
           Returns observations and infos for all agents.
        """

        raise NotImplementedError

    def step(self, actions: Dict) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Execute all drones' actions in parallel.
        Returns obs, rewards, terminations, truncations, infos
        """
        
        raise NotImplementedError

    def render(self):
        """ Visualize current state (optional). """
        pass
    
    def close(self):
        pass

    def init_agent_positions(self):
        """Initialize drone positions at start of episode."""
        pass

    def update_observations(self):
        """ Update observed traffic states. 
            Initially it is filled with the historical average, and will be updated 
            with newly collected data in the episodes.
        """
        pass


class SimBarcaExplore(SimBarcaForecast):
    """
    Dataset that exposes SimBarca traffic simulations for RL exploration.

    The area is divided into grids in `self.grid_ids`, and we assume a drone hovering over a grid can observe all road segments within that grid, and thus obtain almost-perfect traffic states for those segments.

    The data streams are aggregated at two temporal resolutions:
        - High-frequency 5-second aggregates represent drone observations.
        - Low-frequency 3-minute for prediction purpose (in applications we don't need high-frequency predictions)

    The parent class `SimBarcaForecast` loads the flow, density for all sessions as a tensor bundle shaped `[num_sessions, time, num_locations]`.
    Each session behaves like an RL environment rollout, and we want to train our drone policy using all sessions in the training set, and will test it on all sessions in the test set.

    While the parent class assumes at least 30 minutes of historical data (input_window), here we can let the drones fly for 1 step (3 minutes) to collect new data. 
    In this case, the input will have 1 step size data instead of 10 steps, and we let DroneTrafficEnv to do the padding job.  
    The reason of this choice is to avoid designing some non-trivial starting flight plans for the drones (because the predictor needs 10 steps). 

    ``__getitem__`` returns one sliding-window sample, and iterating over ``sample_per_session`` covers a complete pass through the active session.
    """

    # these statistics are calculated using the training split with the `safe_stats` function in debug mode
    # they will be used for both the training and testing set
    data_stats = {
        "5s": {
            "flow": {"mean": 0.188, "std": 0.348},
            "density": {"mean": 0.059, "std": 0.105},
            "speed": {"mean": 5.683, "std": 4.506},
        },
        "3min": {
            "flow": {"mean": 0.192, "std": 0.209},
            "density": {"mean": 0.060, "std": 0.090},
            "speed": {"mean": 5.561, "std": 3.336},
        },
    }


    def __init__(self, split="train", input_window=30, pred_window=30, step_size=3, grid_size=220):
        super().__init__(split, input_window, pred_window, step_size)
        self._active_session = None  # Current session data
        self.grid_size = grid_size

        self.prepare_data_sequences()
        self.load_or_compute_metadata()
        self.clean_up_raw_sequences()

    @property
    def metadata_file(self) -> str:
        return os.path.join(self.metadata_folder, f"{self.__class__.__name__.lower()}.pkl")

    def __len__(self):
        return self.sample_per_session
    
    def __getitem__(self, idx):
        """ Get the idx-th pair of (past, future) data sample from the active session. 
        """
        if self._active_session is None:
            # Return data sample at idx from active session
            raise ValueError("Must set self.active_session first.")
        time_in_day_5s = repeat(
            self.time_in_day_5s[self.in_index[idx]], 
            "t -> t n",
            n=self.veh_flow_5s.shape[-1],
        )
        # stack channels along the last dim
        # past: 5s resolution [Tin, N, 4] => (flow, density, speed, time_in_day)
        past = torch.stack([
            self.veh_flow_5s[self.active_session, self.in_index[idx]],
            self.veh_density_5s[self.active_session, self.in_index[idx]],
            self.veh_speed_5s[self.active_session, self.in_index[idx]],
            time_in_day_5s,
        ], dim=-1)

        # future: low-frequency resolution [Tout, N, 3] => (flow, density, speed)
        future = torch.stack([
            self.veh_flow_3min[self.active_session, self.out_index[idx]],
            self.veh_density_3min[self.active_session, self.out_index[idx]],
            self.veh_speed_3min[self.active_session, self.out_index[idx]],
        ], dim=-1)

        return {"past": past, "future": future}
    
    @property
    def active_session(self):
        """Return current active session data."""
        return self._active_session
    
    @active_session.setter
    def active_session(self, idx):
        """Set current active session by index."""
        self._active_session = idx % self.num_sessions
    
    def iterate_session(self):
        """ Iterate within the current active session.
        """
        if self._active_session is None:
            raise ValueError("Active session not set.")
        for i in range(self.sample_per_session):
            yield i, self.__getitem__(i)
        
    def prepare_data_sequences(self):

        def shift_left(arr, by):
            # shift an 1d array to left by `by` steps, fill in False 
            return np.concatenate([arr[by:], np.full(by, False)])
    
        self.time_in_day_3min = (self._timestamp_3min - self._timestamp_3min[0].astype('datetime64[D]')) / np.timedelta64(24, "h")
        self.time_in_day_5s = (self._timestamp_5s - self._timestamp_5s[0].astype('datetime64[D]')) / np.timedelta64(24, "h")

        # normalize by spatial length and time window; unit: veh/m for flow, veh/m for density
        self.veh_flow_3min = self._vdist_3min / (self.segment_lengths * 180) 
        self.veh_density_3min = self._vtime_3min / (self.segment_lengths * 180)
        self.veh_speed_3min = np.divide(self._vdist_3min, self._vtime_3min)
        # in the raw sequences, the NaN values means there is no vehicle in the road segment during the 
        # time interval. This happens more frequently for the per-5s stats. We can safely replace the NaNs
        # in vehicle flow and density with 0s, as we are not changing the meaning. But we shouldn't do it 
        # for speed, as a 0 speed means vehicles are stopped, not 'no vehicle here'. 
        self.veh_flow_3min = np.nan_to_num(self.veh_flow_3min, nan=0.0)
        self.veh_density_3min = np.nan_to_num(self.veh_density_3min, nan=0.0)

        _, out_index = self.get_sample_in_out_index(self._timestamp_3min)
        shifted_out_index = [
            shift_left(out_index[0, :].copy(), x) 
            for x in reversed(range(1, self.input_window // self.step_size))
        ]
        self.out_index = np.concatenate([np.stack(shifted_out_index, axis=0), out_index], axis=0)
        # for x in range(len(self.out_index)):
        #     nz = np.nonzero(self.out_index[x, :])[0]
        #     print("{} non-zero elements:{}".format(nz.size, nz.tolist()))

        self.veh_flow_5s = self._vdist_5s / (self.segment_lengths * 5)
        self.veh_density_5s = self._vtime_5s / (self.segment_lengths * 5)
        self.veh_speed_5s = np.divide(self._vdist_5s, self._vtime_5s)
        self.veh_flow_5s = np.nan_to_num(self.veh_flow_5s, nan=0.0)
        self.veh_density_5s = np.nan_to_num(self.veh_density_5s, nan=0.0)

        in_index, _ = self.get_sample_in_out_index(self._timestamp_5s)
        steps_per_shift = (self.step_size * 60) // 5  # minutes→5 s ticks
        shifted_in_index = [
            shift_left(in_index[0, :], steps_per_shift * x)
            for x in reversed(range(1, self.input_window // self.step_size))
        ]
        self.in_index = np.concatenate([np.stack(shifted_in_index, axis=0), in_index], axis=0)
        # for x in range(len(self.in_index)):
        #     nz = np.nonzero(self.in_index[x, :])[0]
        #     print("{} non-zero elements:{}".format(nz.size, nz.tolist()))

        self.sample_per_session = self.in_index.shape[0]

        self.veh_flow_3min = torch.from_numpy(self.veh_flow_3min).to(torch.float32)
        self.veh_density_3min = torch.from_numpy(self.veh_density_3min).to(torch.float32)
        self.veh_speed_3min = torch.from_numpy(self.veh_speed_3min).to(torch.float32)

        self.veh_flow_5s = torch.from_numpy(self.veh_flow_5s).to(torch.float32)
        self.veh_density_5s = torch.from_numpy(self.veh_density_5s).to(torch.float32)
        self.veh_speed_5s = torch.from_numpy(self.veh_speed_5s).to(torch.float32)

        self.time_in_day_3min = torch.from_numpy(self.time_in_day_3min).to(torch.float32)
        self.time_in_day_5s = torch.from_numpy(self.time_in_day_5s).to(torch.float32)

        self.in_index = torch.from_numpy(self.in_index.astype(np.bool_)).to(torch.bool)
        self.out_index = torch.from_numpy(self.out_index.astype(np.bool_)).to(torch.bool)


    def load_or_compute_metadata(self):

        def safe_stats(array: np.ndarray) -> Dict[str, float]:
            values = np.asarray(array, dtype=np.float64)
            finite_mask = np.isfinite(values)
            if not np.any(finite_mask):
                return {"mean": float("nan"), "std": float("nan")}
            finite_values = values[finite_mask]
            return {
                "mean": round(float(finite_values.mean()), 3),
                "std": round(float(finite_values.std()), 3),
            }
        
        # coarse grid assignment for spatial abstraction used by the RL agents
        grid_xy = np.floor_divide(self.node_coordinates, self.grid_size).astype(int)
        grid_xy = grid_xy - grid_xy.min(axis=0, keepdims=True)
        grid_width = int(grid_xy[:, 0].max() + 1)
        # grid_height = int(grid_xy[:, 1].max() + 1)
        grid_ids = grid_xy[:, 1] * grid_width + grid_xy[:, 0]
        self.grid_ids = grid_ids

        # self.visualzie_grid_ids(grid_width, grid_height, grid_xy, grid_ids)

        metadata = {
            "adjacency": torch.as_tensor(self.adjacency, dtype=torch.long),
            "edge_index": torch.as_tensor(self.edge_index, dtype=torch.long),
            "node_coordinates": torch.as_tensor(self.node_coordinates, dtype=torch.float32),
            "segment_lengths": torch.as_tensor(self.segment_lengths, dtype=torch.float32),
            "cluster_id": torch.as_tensor(self.cluster_id, dtype=torch.long),
            "grid_id": torch.as_tensor(self.grid_id, dtype=torch.long),
            "grid_xy": torch.as_tensor(grid_xy, dtype=torch.long),
            "num_lanes": torch.as_tensor(self.num_lanes, dtype=torch.long),
            "data_channels": {
                "past": ["flow", "density", "speed", "time"],
                "future": ["flow", "density", "speed"],
            },
            "data_stats": {
                "past": self.data_stats['5s'],
                "future": self.data_stats['3min'],
            },
            "data_null_value": self.data_null_value,
        }

        self.metadata = metadata

        return metadata


    @staticmethod
    def visualzie_grid_ids(grid_width, grid_height, grid_xy, grid_ids):
        # Visualization
        import matplotlib.pyplot as plt

        # Create a grid map showing grid IDs
        grid_map = np.full((grid_height, grid_width), -1, dtype=int)
        for i, (x, y) in enumerate(grid_xy):
            grid_map[int(y), int(x)] = grid_ids[i]

        plt.figure(figsize=(12, 8))

        # Set up the plot
        plt.xlim(-0.5, grid_width - 0.5)
        plt.ylim(-0.5, grid_height - 0.5)

        # Add grid lines
        for x in range(grid_width):
            plt.axvline(x - 0.5, color='gray', linewidth=0.5)
        for y in range(grid_height):
            plt.axhline(y - 0.5, color='gray', linewidth=0.5)

        plt.title('Grid IDs Visualization')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')

        # Add text annotations for grid IDs
        for y in range(grid_height):
            for x in range(grid_width):
                if grid_map[y, x] != -1:
                    plt.text(x, y, str(grid_map[y, x]), ha='center', va='center',fontsize=10, color='black')

        plt.tight_layout()
        plt.savefig('SimBarcaExplore_grid_ids.pdf', bbox_inches='tight')
        plt.close()


class TrafficPredictor:

    def __init__(self, input_steps, output_steps):
        self.prediction_network:nn.Module = None
    
    def predict(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pass


class RewardCalculator:

    def __init__(self):
        pass
    
    def compute(self, predictor: TrafficPredictor, new_data: List) -> float:
        """
        Compute reward based on predictor + newly collected data.
        Could be uncertainty reduction, entropy drop, etc.
        """
        raise NotImplementedError



class MonitoringAgent:
    def __init__(self, obs_space: Any, action_space: Any, config: Dict):
        """Initialize RL policy (PPO, SAC, etc.)."""
        self.policy_net:nn.Module = None
    
    def select_action(self, obs: Dict) -> List[Any]:
        """Choose next drone actions given observation."""
        raise NotImplementedError
    
    def update(self, experiences: List[Tuple]) -> None:
        """Update policy from collected experience (RL training)."""
        raise NotImplementedError


if __name__ == "__main__":
    dataset = SimBarcaExplore(split="train")
    dataset.active_session = 6
    for sample in dataset.iterate_session():
        print(sample)
