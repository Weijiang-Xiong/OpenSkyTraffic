import os
from typing import Dict, List, Tuple

import torch
import numpy as np
from einops import repeat

from skytraffic.data.datasets.simbarca_base import SimBarcaForecast

D_FREQ = 5  # the high-frequency drone data has a time step of 5 seconds
T_STEP = 180  # we require prediction model to predict every 3 minutes (180 seconds)

def _tuple_keys_to_str(d: Dict[Tuple[int, int], int]) -> Dict[str, int]:
    return {f"{k[0]}_{k[1]}": v for k, v in d.items()}

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
        "source": ["flow", "density", "time"],  # "speed",
        "target": ["flow", "density"],  # "speed"
    }
    # data dimensions in the input (excluding auxiliary features like time in day)
    data_dim = [0, 1]
    num_nodes = 1570

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
        augmentations: List = None,
        norm_tid: bool = False,
        vectorized: bool = False,
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
            augmentations" : a list of data augmentation modules to apply during data loading, for supervised training only.
            vectorized: if True, allow `active_session` to be a vector (1D torch.Tensor) for vectorized iteration in `iterate_active_session`. and the data samples from `__getitem__` will retrieve the data sample from all active sessions (with the same time index) and return a batch.
        """
        super().__init__(split, input_window, pred_window, step_size, num_unpadded_samples)
        self._active_session = None  # Current session data
        self.grid_size = grid_size
        self.allow_shorter_input = allow_shorter_input
        self.pad_input = pad_input
        self.norm_tid = norm_tid # whether to normalize time-in-day encoding to have zero mean and unit variance
        # allow active_session to be a vector (1D torch.Tensor) for vectorized iteration in `iterate_active_session`
        self.vectorized = vectorized

        if allow_shorter_input:
            assert input_window // step_size >= 2, (
                f"input_window ({input_window}) should be at least twice of step_size ({step_size}) to allow shorter input."
            )

        self.prepare_data_sequences()
        self.load_or_compute_metadata()
        self.clean_up_raw_sequences()

        self.augmentations = []
        if augmentations is not None:
            if not isinstance(augmentations, list): # single augmentation
                augmentations = [augmentations]
            for aug in augmentations:
                aug.set_grid(self.grid_xy_to_id, self.grid_id)
                self.augmentations.append(aug)

    @property
    def metadata_file(self) -> str:
        return os.path.join(self.metadata_folder, f"{self.__class__.__name__.lower()}.pkl")

    def __len__(self):
        """ the length of dataset for supervised training
        """
        return self.num_sessions * self.sample_per_session

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """ Get the idx-th pair of (past, future) data sample from the active session.
            works for both supervised training and RL data retrieval (through iterate_active_session).
        """

        active_session = (
            self.active_session if self.active_session is not None else (idx // self.sample_per_session)
        )
        sample_idx = idx % self.sample_per_session
        if (self.active_session is not None) and (idx >= self.sample_per_session):
            print("WARNING: idx >= sample_per_session when active_session is set.")
    
        sample_in_index = self.in_index[sample_idx]
        sample_out_index = self.out_index[sample_idx]

        time_in_day_5s = repeat(
            self.time_in_day_5s[self.in_index[sample_idx]],
            "t -> t n",
            n=self.veh_flow_5s.shape[-1],
        )

        if self.vectorized:
            # vectorize over multiple active sessions, requires advanced indexing
            active_session = active_session.unsqueeze(-1)  # shape (B,1)
            sample_in_index = sample_in_index.nonzero(as_tuple=True)[0].unsqueeze(0)  # shape (1, T_session)
            sample_out_index = sample_out_index.nonzero(as_tuple=True)[0].unsqueeze(0)  # shape (1, T_session)
            time_in_day_5s = repeat(time_in_day_5s, "t n -> b t n", b=active_session.shape[0])  # shape (B, Tin, N)

        # stack channels along the last dim
        # past: 5s resolution [Tin, N, 4] => (flow, density, speed, time_in_day), without speed, the feature dimension becomes 3
        source = torch.stack(
            [
                self.veh_flow_5s[active_session, sample_in_index, :],
                self.veh_density_5s[active_session, sample_in_index, :],
                # self.veh_speed_5s[active_session, sample_in_index, :],
                time_in_day_5s,
            ],
            dim=-1,
        )

        # check the vectorized source data matches naive iteration
        # naive_iter_version = torch.stack(
        #     [
        #         torch.stack(
        #             [
        #                 self.veh_flow_5s[x, sample_in_index.squeeze(), :],
        #                 self.veh_density_5s[x, sample_in_index.squeeze(), :],
        #                 # self.veh_speed_5s[x, sample_in_index, :],
        #                 time_in_day_5s[0, :, :],
        #             ],
        #             dim=-1,
        #         )
        #         for x in active_session
        #     ]
        # )
        # assert torch.allclose(source, naive_iter_version), "Vectorized source data does not match naive iteration."

        # future: low-frequency resolution [Tout, N, 3] => (flow, density, speed)
        target = torch.stack(
            [
                self.veh_flow_3min[active_session, sample_out_index, :],
                self.veh_density_3min[active_session, sample_out_index, :],
                # self.veh_speed_3min[active_session, sample_out_index, :],
            ],
            dim=-1,
        )

        # if the observation window is smaller than input_window, pad it with mean values
        # this happens at the session begeinning because the drone just start to collect traffic data.
        T_in = source.shape[1] if self.vectorized else source.shape[0]
        if self.pad_input and T_in < self.input_size[0]:
            source, temporal_padding_mask = self.pad_backward_time(source, return_mask=True)
        else:
            # no padding applied, all time steps have valid observations (though not all locations are observed)
            temporal_padding_mask = torch.ones(source.shape[0]).bool()

        data_dict = {"source": source, "target": target}
        if self.pad_input:
            data_dict["temporal_padding_mask"] = temporal_padding_mask

        return data_dict

    def pad_backward_time(self, source: torch.Tensor, pad_len=None, zero_pad=False, return_mask=False) -> torch.Tensor:
        """ 
        Pad the input data backward in time with global mean values.
        When the input data has less time steps (e.g., 1 step, 3 min) than the required input window (e.g., 30 min)
        we pad the beginning with global mean values to reach the required input window size, and we calculate the 
        time-in-day encodings by linear extrapolation.
        
        Args:
            pad_len: if specified, pad by this length, otherwise pad to the full input window size.
            zero_pad: if True, pad with zeros instead of global mean values.
            return_mask: if True, also return a temporal padding mask indicating which time steps are padded (True for real data, False for padded data).
        """
        if self.vectorized:
            N, T, P, C = source.shape
            time_enc = source[0, :, 0, -1]
        else:
            T, P, C = source.shape
            time_enc = source[:, 0, -1]

        start_time_enc = time_enc[0]  
        dt = time_enc.diff().abs().mean()  # average time difference between consecutive time steps

        pad_len = self.input_size[0] - T if pad_len is None else pad_len

        pad_data = torch.ones((pad_len, P, C-1)) * (
                torch.tensor(
                    [
                        self.data_stats["5s"]["flow"]["mean"] if not zero_pad else 0.0,
                        self.data_stats["5s"]["density"]["mean"] if not zero_pad else 0.0,
                        # self.data_stats['5s']['speed']['mean'] if not zero_pad else 0.0,
                    ]
                ).reshape(1, 1, -1)
            )
        
        # count back dt per step until the input window is reached: -pad_len*dt, ... -2 *dt, -dt
        count_back_time = ( torch.arange(- pad_len, 0, step=1) ) * dt
        count_back_time = start_time_enc + repeat(count_back_time, "T -> T P", P=P)
        pad = torch.cat([pad_data, count_back_time.unsqueeze(-1)], dim=-1)

        if self.vectorized:
            pad = repeat(pad, "T P C -> N T P C", N=N)

        temporal_padding_mask = torch.cat(tensors=[torch.zeros(pad_len), torch.ones(T)], dim=0).bool()
        padded_source = torch.cat([pad, source], dim=1 if self.vectorized else 0)

        if return_mask:
            return padded_source, temporal_padding_mask
        else:
            return padded_source

    def collate_fn(self, list_of_data_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if self.vectorized:
            raise ValueError("collate_fn should not be used in vectorized mode.")
        
        batch_data_dict = {
            key: torch.stack([data_dict[key] for data_dict in list_of_data_dicts], dim=0)
            for key in list_of_data_dicts[0].keys()
        }

        for augmentor in self.augmentations:
            batch_data_dict = augmentor(batch_data_dict)

        return batch_data_dict

    @property
    def active_session(self):
        """Return current active session data."""
        return self._active_session

    @active_session.setter
    def active_session(self, idx):
        """Set current active session by index."""
        if not self.vectorized:
            try:
                idx = int(idx)
            except Exception as e:
                raise ValueError(f"Active session should be an integer for non-vectorized mode: {e}")
        else:
            try:
                idx = torch.as_tensor(idx, dtype=torch.long)
            except Exception as e:
                raise ValueError(f"Active session should be a 1D torch.Tensor for vectorized mode: {e}")
        self._active_session = idx % self.num_sessions # this also works for interger tensors

    def iterate_active_session(self):
        """Iterate within the current active session."""
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
        # the simulation starts from 8am and ends roughly at 10 am, so the time in day values are uniformly distributed
        # between 8/24 and 10/24, we can normalize them to have zero mean and unit variance for better training stability
        # we do it using prior knowledge of the time range instead of calculating mean and std from data
        if self.norm_tid: 
            self.time_in_day_3min = np.sqrt(3) * (24 * self.time_in_day_3min - 9) 
            self.time_in_day_5s = np.sqrt(3) * (24 * self.time_in_day_5s - 9)

        # normalize by spatial length and time window; unit: veh/m for flow, veh/m for density
        self.veh_flow_3min = self._vdist_3min / (self.segment_lengths * T_STEP)
        self.veh_density_3min = self._vtime_3min / (self.segment_lengths * T_STEP)
        # self.veh_speed_3min = np.divide(self._vdist_3min, self._vtime_3min)
        # in the raw sequences, the NaN values means there is no vehicle in the road segment during the
        # time interval. This happens more frequently for the per-5s stats. We can safely replace the NaNs
        # in vehicle flow and density with 0s, as we are not changing the meaning. But we shouldn't do it
        # for speed, as a 0 speed means vehicles are stopped, not 'no vehicle here'.
        self.veh_flow_3min = np.nan_to_num(self.veh_flow_3min, nan=0.0)
        self.veh_density_3min = np.nan_to_num(self.veh_density_3min, nan=0.0)

        self.veh_flow_5s = self._vdist_5s / (self.segment_lengths * D_FREQ)
        self.veh_density_5s = self._vtime_5s / (self.segment_lengths * D_FREQ)
        # self.veh_speed_5s = np.divide(self._vdist_5s, self._vtime_5s)
        self.veh_flow_5s = np.nan_to_num(self.veh_flow_5s, nan=0.0)
        self.veh_density_5s = np.nan_to_num(self.veh_density_5s, nan=0.0)

        # input and output indexes for sliding window sampling
        in_index, _ = self.get_sample_in_out_index(self._timestamp_5s)
        _, out_index = self.get_sample_in_out_index(self._timestamp_3min)

        # shift the boolean mask to the left to allow shorter input windows at the beginning of each session
        # e.g., if input_window=30min, step_size=3min, then we allow input windows of size 3,6,9,...,27 min
        if self.allow_shorter_input:
            shifted_out_index = [
                shift_left(out_index[0, :].copy(), x)
                for x in reversed(range(1, self.input_window // self.step_size))
            ]
            out_index = np.concatenate([np.stack(shifted_out_index, axis=0), out_index], axis=0)

            steps_per_shift = int((self.step_size * 60) // 5)  # minutes→5 s ticks
            shifted_in_index = [
                shift_left(in_index[0, :], steps_per_shift * x)
                for x in reversed(range(1, self.input_window // self.step_size))
            ]
            in_index = np.concatenate([np.stack(shifted_in_index, axis=0), in_index], axis=0)

            # check the time alignment between input and output indexes
            # the input have higher frequency than output, so we need to scale the input index
            # by the steps_per_shift factor
            # for sii, soi in zip(in_index, out_index):
            #     last_in = np.nonzero(sii)[0][-1]
            #     first_out = np.nonzero(soi)[0][0]
            #     assert (last_in // steps_per_shift + 1) == first_out, "Input index last True should be before output index first True."

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

    def _compute_coverage_mask(self, positions: List[Tuple[int, int]]) -> np.ndarray:
        """
        compute a boolean mask of shape (num_locations,) indicating which locations are covered
        at the drones at current `positions`
        """
        # simply add up the per-drone coverage masks.
        # If a location is covered by at least one drone, it is considered covered.
        mask = sum([self.grid_id == self.grid_xy_to_id[(x, y)] for x, y in positions])

        return (mask > 0).astype(np.int8)

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
        grid_xy_to_id_np = np.unique(np.concatenate([grid_xy, grid_id[:, None]], axis=1), axis=0)
        self.grid_xy_to_id = {(int(x), int(y)): int(gid) for x, y, gid in grid_xy_to_id_np}

        # self.visualzie_grid_ids(grid_width, grid_height, grid_xy, grid_ids)

        # input and output sizes (excluding batch dimension)
        self.input_size = (self.input_window * 12, self.adjacency.shape[0], len(self.data_channels["source"]))
        self.output_size = (self.pred_window // self.step_size, self.adjacency.shape[0], len(self.data_channels["target"]))

        metadata = {
            "adjacency": torch.as_tensor(self.adjacency, dtype=torch.long),
            "edge_index": torch.as_tensor(self.edge_index, dtype=torch.long),
            "node_coordinates": torch.as_tensor(self.node_coordinates, dtype=torch.float32),
            "segment_lengths": torch.as_tensor(self.segment_lengths, dtype=torch.float32),
            "cluster_id": torch.as_tensor(self.cluster_id, dtype=torch.long),
            # the grid ID of each road segment
            "grid_id": torch.as_tensor(self.grid_id, dtype=torch.long),  
            # the grid (x, y) coordinate of each road segment, shape (N,2)
            "grid_xy": torch.as_tensor(grid_xy, dtype=torch.long),
            # mapping from grid (x, y) coordinate to grid ID
            # converted to string keys for omegaconf compatibility
            "grid_xy_to_id": _tuple_keys_to_str(self.grid_xy_to_id),  
            "num_lanes": torch.as_tensor(self.num_lanes, dtype=torch.long),
            "data_stats": {
                "source": self.data_stats["5s"],
                "target": self.data_stats["3min"],
            },
            "data_channels": self.data_channels,
            "data_dim": self.data_dim,
            "data_null_value": self.data_null_value,
            "input_size": self.input_size,  # 5s resolution
            "output_size": self.output_size,  # 3min resolution
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
            plt.axvline(x - 0.5, color="gray", linewidth=0.5)
        for y in range(grid_height):
            plt.axhline(y - 0.5, color="gray", linewidth=0.5)

        plt.title("Grid IDs Visualization")
        plt.xlabel("Grid X")
        plt.ylabel("Grid Y")

        # Add text annotations for grid IDs
        for y in range(grid_height):
            for x in range(grid_width):
                if grid_map[y, x] != -1:
                    plt.text(
                        x,
                        y,
                        str(grid_map[y, x]),
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="black",
                    )

        plt.tight_layout()
        plt.savefig("SimBarcaExplore_grid_ids.pdf", bbox_inches="tight")
        plt.close()

    def summarize(self):
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Use seaborn darkgrid style
        sns.set_theme(style="darkgrid")

        # Draw a histogram for the 3 min flow and density values separately,
        # and a scatter plot for flow vs density
        flow_values = self.veh_flow_3min.numpy().flatten()
        density_values = self.veh_density_3min.numpy().flatten()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].hist(flow_values, bins=50, color='blue', alpha=0.7)
        axes[0].set_title('3-min Flow Distribution')
        axes[0].set_xlabel('Flow (veh/s)')
        axes[0].set_ylabel('Data point count')

        axes[1].hist(density_values, bins=50, color='green', alpha=0.7)
        axes[1].set_title('3-min Density Distribution')
        axes[1].set_xlabel('Density (veh/m)')
        axes[1].set_ylabel('Data point count')

        axes[2].scatter(np.average(self.veh_density_3min.numpy(), weights=self.segment_lengths, axis=-1).flatten(),
                        np.average(self.veh_flow_3min.numpy(), weights=self.segment_lengths, axis=-1).flatten(),
                        alpha=0.5) 
        axes[2].set_title('Regional Avg Flow vs Density')
        axes[2].set_xlabel('Density (veh/m)')
        axes[2].set_ylabel('Flow (veh/s)')

        fig.tight_layout()
        fig.savefig('SimBarcaExplore_{}set_summary.pdf'.format(self.split))
        plt.close(fig)

        

if __name__ == "__main__":
    trainset = SimBarcaExplore(
        split="train",
        input_window=30,
        pred_window=30,
        step_size=3,
        num_unpadded_samples=20,
        allow_shorter_input=True,
        pad_input=True,
        vectorized=False
    )
    # trainset.summarize()

    # supervised training data loading
    for k, v in trainset.metadata.items():
        if isinstance(v, (torch.Tensor, np.ndarray)):
            print(f"{k}: {v.shape}")
        else:
            print(f"{k}: {v}")
    print("Number of samples per session:", trainset.sample_per_session)
    print("Total number of samples:", len(trainset))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, collate_fn=trainset.collate_fn)
    for batch in train_loader:
        print(batch["source"].shape, batch["target"].shape)
        print("\n")
        break

    # RL exploration data loading
    trainset.active_session = 7  # set the active session
    for step, new_data in trainset.iterate_active_session():
        print(step, new_data["source"].shape, new_data["target"].shape)
        if step >= 5:
            break
    
    # test vectorized iteration
    print("Testing vectorized iteration...")
    trainset.vectorized = True
    trainset.active_session = torch.tensor([0,1,2,3])
    for idx, data in trainset.iterate_active_session():
        print("Index:", idx, "Data source shape:", data["source"].shape)
        if idx >= 5:
            break

    
    print("a special case of running vectorized mode with only one active session...")
    trainset.active_session = torch.tensor([4])
    for idx, data in trainset.iterate_active_session():
        print("Index:", idx, "Data source shape:", data["source"].shape)
        if idx >= 5:
            break


    testset = SimBarcaExplore(
        split="test",
        input_window=30,
        pred_window=30,
        step_size=3,
        num_unpadded_samples=20,
        allow_shorter_input=True,
        pad_input=True,
        vectorized=False
    )

    # evaluate trivial average value prediction
    from skymonitor.simbarca_explore_evaluation import SimBarcaExploreEvaluator
    evaluator = SimBarcaExploreEvaluator(
        save_dir="scratch/simbarca_explore_baseline_results",
        visualize=False, 
        ignore_value=0.0
        )
    
    # eval historical average prediction
    density = trainset.veh_density_3min.numpy() # shape (num_session, time, num_location)
    flow = trainset.veh_flow_3min.numpy() # shape (num_session, time, num_location)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, collate_fn=testset.collate_fn)
    _, all_labels = evaluator.collect_predictions(model=torch.nn.Identity(), data_loader=test_loader, pred_seqs=[], data_seqs=['target'])
    target = all_labels['target']  # shape (num_samples, T, P, C)
    N, T, P, C = target.shape

    global_average_prediction = repeat(
        torch.as_tensor([flow.mean(), density.mean()], dtype=torch.float32),
        "C -> N T P C", N=N, T=T, P=P, C=C
    )
    global_avg_res = evaluator.calculate_error_metrics(global_average_prediction, target, data_channels=['flow', 'density'])
    print("Global Average Prediction Results:")
    for key, value in global_avg_res.items():
        print(f"  {key}: {value}")
    
    per_location_average_prediction = repeat(
        torch.as_tensor([flow.mean(axis=(0,1)), density.mean(axis=(0,1))], dtype=torch.float32),
        "C P -> N T P C", N=N, T=T, P=P, C=C
    )
    per_location_avg_res = evaluator.calculate_error_metrics(per_location_average_prediction, target, data_channels=['flow', 'density'])
    print("Per-Location Average Prediction Results:")
    for key, value in per_location_avg_res.items():
        print(f"  {key}: {value}")

    # average over the simulation sessions. 
    per_location_per_time_avg_flow = repeat(flow.mean(axis=0), "t P -> S t P", S=testset.num_sessions)  # shape (time, P)
    per_location_per_time_avg_density = repeat(density.mean(axis=0), "t P -> S t P", S=testset.num_sessions)  # shape (time, P)
    from copy import deepcopy
    testset_cp = deepcopy(testset)
    testset_cp.veh_flow_3min = torch.as_tensor(per_location_per_time_avg_flow, dtype=torch.float32)
    testset_cp.veh_density_3min = torch.as_tensor(per_location_per_time_avg_density, dtype=torch.float32)

    dataloader_cp = torch.utils.data.DataLoader(testset_cp, batch_size=8, shuffle=False, collate_fn=testset_cp.collate_fn)
    _, avg_preds = evaluator.collect_predictions(model=torch.nn.Identity(), data_loader=dataloader_cp, pred_seqs=[], data_seqs=['target'])
    per_location_per_time_avg_prediction = avg_preds['target']  # shape (num_samples, T, P, C)

    per_location_per_time_avg_res = evaluator.calculate_error_metrics(per_location_per_time_avg_prediction, target, data_channels=['flow', 'density'])
    print("Per-Location-Per-Time Average Prediction Results:")
    for key, value in per_location_per_time_avg_res.items():
        print(f"  {key}: {value}")
