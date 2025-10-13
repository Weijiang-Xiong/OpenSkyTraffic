import os
from typing import Dict, List

import torch
import numpy as np
from einops import repeat

from skytraffic.data.datasets.simbarca_base import SimBarcaForecast

class SimBarcaExplore(SimBarcaForecast):
    """
    Dataset that exposes SimBarca traffic simulations for prediction-in-the-loop RL exploration.

    The area is divided into grids in `self.grid_ids`, and we assume a drone hovering over a grid can observe all road segments within that grid, and thus obtain almost-perfect traffic states for those segments.

    The data streams are aggregated at two temporal resolutions:
        - High-frequency 5-second aggregates represent drone observations.
        - Low-frequency 3-minute for prediction purpose (in applications we don't need high-frequency predictions)

    The parent class `SimBarcaForecast` loads the flow, density for all sessions as a tensor bundle shaped `[num_sessions, time, num_locations]`.
    Each session behaves like an RL environment rollout, and we want to train our drone policy using all sessions in the training set, and will test it on all sessions in the test set.

    While the parent class assumes at least 30 minutes of historical data (`input_window`), here we can let the drones fly for one  `step_size` (3 minutes) to collect new data.
    In the prediction-in-the-loop RL, the input will have 1 step size data instead of 10 steps, and we let the predictor wrapper to do the padding job.
    The reason of this choice is to avoid designing some non-trivial starting flight plans for the drones (because the predictor needs 10 steps).

    In reinforcement learning, the dataset returns data samples from the current active session only.
    Each step, the dataset returns the data in most recent time window of `step_size` (e.g., 3 min).
    We first set `self.active_session` and then use `self.iterate_active_session` to go through all samples in the current active session.

    Still, the `__len__` and `__getitem__` methods are implemented to support supervised training for the prediction model.
    The dataset gives sliding-window samples with fixed input and output window sizes.

    See below for code examples.
    """

    # these statistics are calculated using the training split with the `safe_stats` function in debug mode
    # they will be used for both the training and testing set
    data_stats = {
        "5s": {
            "flow": {"mean": 0.188, "std": 0.348},
            "density": {"mean": 0.059, "std": 0.105},
            # "speed": {"mean": 5.683, "std": 4.506},
        },
        "3min": {
            "flow": {"mean": 0.192, "std": 0.209},
            "density": {"mean": 0.060, "std": 0.090},
            # "speed": {"mean": 5.561, "std": 3.336},
        },
    }
    data_channels = {
        "source": ["flow", "density", "time"], # "speed",
        "target": ["flow", "density"],  # "speed"
    }


    def __init__(
        self,
        split="train",
        input_window=30,
        pred_window=30,
        step_size=3,
        num_unpadded_samples=20,
        grid_size=220,
        allow_shorter_input=True,
        pad_input=True,
    ):
        """
        Args:
            split: "train", "val", or "test"
            input_window: input window size in minutes (multiple of step_size)
            pred_window: prediction window size in minutes (multiple of step_size)
            step_size: step size in minutes for sliding window sampling
            grid_size: grid size in meters for spatial abstraction
            num_unpadded_samples: the number of unpadded samples to extract from the parent class.
            allow_shorter_input: if True, allow the input window to be `1*step_size` at the beginning of each session, and increase by step_size each step until reaching input_window size. False means the input window is always input_window size.
            pad_input: if both `allow_shorter_input` and `pad_input` are True, the input will be padded with global average values to reach the `input_window` size. If `allow_shorter_input` is True but `pad_input` is False, the input will be of variable length.
        """
        super().__init__(split, input_window, pred_window, step_size, num_unpadded_samples)
        self._active_session = None  # Current session data
        self.grid_size = grid_size
        self.allow_shorter_input = allow_shorter_input
        self.pad_input = pad_input

        if allow_shorter_input:
            assert input_window // step_size >= 2, (
                f"input_window ({input_window}) should be at least twice of step_size ({step_size}) to allow shorter input."
            )
        

        self.prepare_data_sequences()
        self.load_or_compute_metadata()
        self.clean_up_raw_sequences()

    @property
    def metadata_file(self) -> str:
        return os.path.join(self.metadata_folder, f"{self.__class__.__name__.lower()}.pkl")

    def __len__(self):
        return self.num_sessions * self.sample_per_session
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """ Get the idx-th pair of (past, future) data sample from the active session. 
        """

        active_session = self.active_session if self.active_session is not None else (idx // self.sample_per_session)
        sample_idx = idx % self.sample_per_session
        if (self.active_session is not None) and (idx >= self.sample_per_session):
            print("WARNING: idx >= sample_per_session when active_session is set.")

        time_in_day_5s = repeat(
            self.time_in_day_5s[self.in_index[sample_idx]], 
            "t -> t n",
            n=self.veh_flow_5s.shape[-1],
        )
        # stack channels along the last dim
        # past: 5s resolution [Tin, N, 4] => (flow, density, speed, time_in_day), without speed, the feature dimension becomes 3
        source = torch.stack([
            self.veh_flow_5s[active_session, self.in_index[sample_idx]],
            self.veh_density_5s[active_session, self.in_index[sample_idx]],
            # self.veh_speed_5s[active_session, self.in_index[sample_idx]],
            time_in_day_5s,
        ], dim=-1)

        # future: low-frequency resolution [Tout, N, 3] => (flow, density, speed)
        target = torch.stack([
            self.veh_flow_3min[active_session, self.out_index[sample_idx]],
            self.veh_density_3min[active_session, self.out_index[sample_idx]],
            # self.veh_speed_3min[active_session, self.out_index[sample_idx]],
        ], dim=-1)

        # if the observation window is smaller than input_window, pad it with mean values
        # this happens at the session begeinning because the drone just start to collect traffic data. 
        if self.pad_input and source.shape[0] < self.metadata["input_size"][0]:
            pad_len = self.metadata["input_size"][0] - source.shape[0]
            pad_data = torch.ones((pad_len, source.shape[1], source.shape[2] - 1)) * (
                torch.tensor([
                    self.data_stats['5s']['flow']['mean'],
                    self.data_stats['5s']['density']['mean'],
                    # self.data_stats['5s']['speed']['mean'],
                ]).reshape(1, 1, -1)
            )
            # count back 5s per step until the input window is reached
            count_back_seconds = torch.arange(0, pad_len*5, step=5) - pad_len*5 #  -pad_len*5, ... -10 ,-5
            pad_time = count_back_seconds / (24*60*60.0)  # seconds in a day
            pad_time = time_in_day_5s[0,0] - repeat(pad_time, "t -> t n", n=source.shape[1])
            pad = torch.cat([pad_data, pad_time.unsqueeze(-1)], dim=-1)

            source = torch.cat([pad, source], dim=0)

        return {"source": source, "target": target}
    
    def collate_fn(self, list_of_data_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            "source": torch.stack([data_dict["source"] for data_dict in list_of_data_dicts], dim=0),
            "target": torch.stack([data_dict["target"] for data_dict in list_of_data_dicts], dim=0),
        }
    
    @property
    def active_session(self):
        """Return current active session data."""
        return self._active_session
    
    @active_session.setter
    def active_session(self, idx):
        """Set current active session by index."""
        self._active_session = idx % self.num_sessions
    
    def iterate_active_session(self):
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
    
        self.time_in_day_3min = self.compute_time_in_day(self._timestamp_3min)
        self.time_in_day_5s = self.compute_time_in_day(self._timestamp_5s)

        # normalize by spatial length and time window; unit: veh/m for flow, veh/m for density
        self.veh_flow_3min = self._vdist_3min / (self.segment_lengths * 180) 
        self.veh_density_3min = self._vtime_3min / (self.segment_lengths * 180)
        # self.veh_speed_3min = np.divide(self._vdist_3min, self._vtime_3min)
        # in the raw sequences, the NaN values means there is no vehicle in the road segment during the 
        # time interval. This happens more frequently for the per-5s stats. We can safely replace the NaNs
        # in vehicle flow and density with 0s, as we are not changing the meaning. But we shouldn't do it 
        # for speed, as a 0 speed means vehicles are stopped, not 'no vehicle here'. 
        self.veh_flow_3min = np.nan_to_num(self.veh_flow_3min, nan=0.0)
        self.veh_density_3min = np.nan_to_num(self.veh_density_3min, nan=0.0)

        self.veh_flow_5s = self._vdist_5s / (self.segment_lengths * 5)
        self.veh_density_5s = self._vtime_5s / (self.segment_lengths * 5)
        # self.veh_speed_5s = np.divide(self._vdist_5s, self._vtime_5s)
        self.veh_flow_5s = np.nan_to_num(self.veh_flow_5s, nan=0.0)
        self.veh_density_5s = np.nan_to_num(self.veh_density_5s, nan=0.0)

        # input and output indexes for sliding window sampling
        in_index, _ = self.get_sample_in_out_index(self._timestamp_5s)
        _, out_index = self.get_sample_in_out_index(self._timestamp_3min)

        if self.allow_shorter_input:
            shifted_out_index = [
                shift_left(out_index[0, :].copy(), x) 
                for x in reversed(range(1, self.input_window // self.step_size))
            ]
            out_index = np.concatenate([np.stack(shifted_out_index, axis=0), out_index], axis=0)

            steps_per_shift = (self.step_size * 60) // 5  # minutes→5 s ticks
            shifted_in_index = [
                shift_left(in_index[0, :], steps_per_shift * x)
                for x in reversed(range(1, self.input_window // self.step_size))
            ]
            in_index = np.concatenate([np.stack(shifted_in_index, axis=0), in_index], axis=0)

        self.in_index = in_index
        self.out_index = out_index
        self.sample_per_session = self.in_index.shape[0]

        # for x in range(len(self.out_index)):
        #     nz = np.nonzero(self.out_index[x, :])[0]
        #     print("{} non-zero elements:{}".format(nz.size, nz.tolist()))
        # for x in range(len(self.in_index)):
        #     nz = np.nonzero(self.in_index[x, :])[0]
        #     print("{} non-zero elements:{}".format(nz.size, nz.tolist()))

        self.veh_flow_3min = torch.from_numpy(self.veh_flow_3min).to(torch.float32)
        self.veh_density_3min = torch.from_numpy(self.veh_density_3min).to(torch.float32)
        # self.veh_speed_3min = torch.from_numpy(self.veh_speed_3min).to(torch.float32)

        self.veh_flow_5s = torch.from_numpy(self.veh_flow_5s).to(torch.float32)
        self.veh_density_5s = torch.from_numpy(self.veh_density_5s).to(torch.float32)
        # self.veh_speed_5s = torch.from_numpy(self.veh_speed_5s).to(torch.float32)

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
        self.grid_xy = grid_xy
        grid_width = int(grid_xy[:, 0].max() + 1)
        grid_height = int(grid_xy[:, 1].max() + 1)
        
        grid_id = grid_xy[:, 1] * grid_width + grid_xy[:, 0]
        self.grid_id = grid_id
        grid_xy_to_id_np = np.unique(
            np.concatenate([grid_xy, grid_id[:, None]], axis=1), axis=0
        )
        self.grid_xy_to_id = {(int(x), int(y)): int(gid) for x, y, gid in grid_xy_to_id_np}

        # self.visualzie_grid_ids(grid_width, grid_height, grid_xy, grid_ids)

        metadata = {
            "adjacency": torch.as_tensor(self.adjacency, dtype=torch.long),
            "edge_index": torch.as_tensor(self.edge_index, dtype=torch.long),
            "node_coordinates": torch.as_tensor(self.node_coordinates, dtype=torch.float32),
            "segment_lengths": torch.as_tensor(self.segment_lengths, dtype=torch.float32),
            "cluster_id": torch.as_tensor(self.cluster_id, dtype=torch.long),
            "grid_id": torch.as_tensor(self.grid_id, dtype=torch.long), # the grid ID of each road segment
            "grid_xy": torch.as_tensor(grid_xy, dtype=torch.long), # the grid (x, y) coordinate of each road segment
            "grid_xy_to_id": self.grid_xy_to_id, # mapping from grid (x, y) coordinate to grid ID
            "num_lanes": torch.as_tensor(self.num_lanes, dtype=torch.long),
            "data_stats": {
                "source": self.data_stats['5s'],
                "target": self.data_stats['3min'],
            },
            "data_null_value": self.data_null_value,
            "input_size": (self.input_window * 12, self.adjacency.shape[0], len(self.data_channels['source'])),  # 5s resolution
            "output_size": (self.pred_window // self.step_size, self.adjacency.shape[0], len(self.data_channels['target'])),  # 3min resolution
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

if __name__ == "__main__":
    dataset = SimBarcaExplore(
        split="train",
        input_window=30,
        pred_window=30,
        step_size=3,
        num_unpadded_samples=20,
        allow_shorter_input=True,
        pad_input=True,
    )
    # supervised training data loading
    for k, v in dataset.metadata.items():
        if isinstance(v, (torch.Tensor, np.ndarray)):
            print(f"{k}: {v.shape}")
        else:
            print(f"{k}: {v}")
    print("Number of samples per session:", dataset.sample_per_session)
    print("Total number of samples:", len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    for batch in dataloader:
        print(batch["source"].shape, batch["target"].shape)
        print("\n")
        break

    # RL exploration data loading
    dataset.active_session = 7  # set the active session
    for step, new_data in dataset.iterate_active_session():
        print(step, new_data["source"].shape, new_data["target"].shape)
        if step >= 5:
            break