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
    Traffic dataset for Barcelona road network with speed and flow for each road segment.
    The data streams are aggregated at two temporal resolutions:
        - Low-frequency 3-minute aggregates for observation and prediction.
        - High-frequency 5-second aggregates are available but not used for input.
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
        grid_size=220,
        norm_tid: bool = False,
        augmentations: List = None,
    ):
        """
        Args:
            split: "train", "val", or "test"
            input_window: input window size in minutes (multiple of step_size)
            pred_window: prediction window size in minutes (multiple of step_size)
            step_size: step size in minutes for sliding window sampling
            grid_size: grid size in meters for spatial abstraction
            norm_tid: whether to normalize time-of-day encoding to have zero mean and unit variance
            augmentations" : a list of data augmentation modules to apply during data loading, for supervised training only.
        """
        # the number of sliding window samples during the 120 minute valid simulation
        num_unpadded_samples = ( 120 - (input_window + pred_window) ) // step_size
        super().__init__(split, input_window, pred_window, step_size, num_unpadded_samples)
        self.grid_size = grid_size
        self.norm_tid = norm_tid # whether to normalize time-in-day encoding to have zero mean and unit variance

        self.prepare_data_sequences()
        self.load_or_compute_metadata()
        self.clean_up_raw_sequences()

        if augmentations is not None:
            if not isinstance(augmentations, list): # single augmentation
                augmentations = [augmentations]
            for aug in augmentations:
                aug.set_grid(self.grid_xy_to_id, self.grid_id)
            self.augmentations = augmentations
        else:
            self.augmentations = []

    @property
    def metadata_file(self) -> str:
        return os.path.join(self.metadata_folder, f"{self.__class__.__name__.lower()}.pkl")

    def __len__(self):
        """ the length of dataset for supervised training
        """
        return self.num_sessions * self.sample_per_session

    def __getitem__(self, idx: int | Tuple[int, int]) -> Dict[str, torch.Tensor]:
        """ Get the idx-th pair of (past, future) data sample from the active session. """

        if isinstance(idx, tuple):
            session_idx, sample_idx = idx
        else:
            session_idx = idx // self.sample_per_session
            sample_idx = idx % self.sample_per_session

        assert session_idx >= 0 and session_idx < self.num_sessions, "Session index out of range."
        assert sample_idx >= 0 and sample_idx < self.sample_per_session, "Sample index out of range."

        sample_in_index = self.in_index[sample_idx]
        sample_out_index = self.out_index[sample_idx]

        time_in_day_3min = np.asarray(self.time_in_day_3min[sample_in_index])
        time_in_day_3min = repeat(time_in_day_3min, "t -> t n", n=self.veh_flow_3min.shape[-1])

        # stack channels along the last dim
        # past: 3-min resolution [Tin, N, 4] => (flow, density, speed, time_in_day), without speed, the feature dimension becomes 3
        source = np.stack(
            [
                np.asarray(self.veh_flow_3min[session_idx, sample_in_index, :]),
                np.asarray(self.veh_density_3min[session_idx, sample_in_index, :]),
                # self.veh_speed_3min[session_idx, sample_in_index, :],
                time_in_day_3min,
            ],
            axis=-1,
        )

        # future: low-frequency resolution [Tout, N, 3] => (flow, density, speed)
        target = np.stack(
            [
                np.asarray(self.veh_flow_3min[session_idx, sample_out_index, :]),
                np.asarray(self.veh_density_3min[session_idx, sample_out_index, :]),
                # self.veh_speed_3min[session_idx, sample_out_index, :],
            ],
            axis=-1,
        )

        return {
            "source": torch.as_tensor(source, dtype=torch.float32),
            "target": torch.as_tensor(target, dtype=torch.float32),
        }

    def collate_fn(self, list_of_data_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch_data_dict = {
            key: torch.stack([data_dict[key] for data_dict in list_of_data_dicts], dim=0)
            for key in list_of_data_dicts[0].keys()
        }

        for augmentor in self.augmentations:
            batch_data_dict = augmentor(batch_data_dict)

        return batch_data_dict

    def prepare_data_sequences(self):
        self.time_in_day_3min = self.compute_time_in_day(self._timestamp_3min)
        self.time_in_day_5s = self.compute_time_in_day(self._timestamp_5s)
        # the simulation starts from 8am and ends roughly at 10 am, so the time in day values are uniformly distributed
        # between 8/24 and 10/24, we can normalize them to have zero mean and unit variance for better training stability
        # we do it using prior knowledge of the time range instead of calculating mean and std from data
        if self.norm_tid: 
            self.time_in_day_3min = np.sqrt(3) * (24 * self.time_in_day_3min - 9) 
            self.time_in_day_5s = np.sqrt(3) * (24 * self.time_in_day_5s - 9)
        self.time_in_day_3min = np.asarray(self.time_in_day_3min, dtype=np.float32)
        self.time_in_day_5s = np.asarray(self.time_in_day_5s, dtype=np.float32)

        # normalize by spatial length and time window; unit: veh/m for flow, veh/m for density
        self.veh_flow_3min = self._vdist_3min / (self.segment_lengths * T_STEP)
        self.veh_density_3min = self._vtime_3min / (self.segment_lengths * T_STEP)
        # self.veh_speed_3min = np.divide(self._vdist_3min, self._vtime_3min)
        # in the raw sequences, the NaN values means there is no vehicle in the road segment during the
        # time interval. This happens more frequently for the per-5s stats. We can safely replace the NaNs
        # in vehicle flow and density with 0s, as we are not changing the meaning. But we shouldn't do it
        # for speed, as a 0 speed means vehicles are stopped, not 'no vehicle here'.
        self.veh_flow_3min = np.nan_to_num(self.veh_flow_3min, nan=0.0).astype(np.float32, copy=False)
        self.veh_density_3min = np.nan_to_num(self.veh_density_3min, nan=0.0).astype(np.float32, copy=False)

        self.veh_flow_5s = self._vdist_5s / (self.segment_lengths * D_FREQ)
        self.veh_density_5s = self._vtime_5s / (self.segment_lengths * D_FREQ)
        # self.veh_speed_5s = np.divide(self._vdist_5s, self._vtime_5s)
        self.veh_flow_5s = np.nan_to_num(self.veh_flow_5s, nan=0.0).astype(np.float32, copy=False)
        self.veh_density_5s = np.nan_to_num(self.veh_density_5s, nan=0.0).astype(np.float32, copy=False)

        # input and output indexes for sliding window sampling
        in_index, _ = self.get_sample_in_out_index(self._timestamp_3min)
        _, out_index = self.get_sample_in_out_index(self._timestamp_3min)

        self.in_index = in_index
        self.out_index = out_index
        self.sample_per_session = self.in_index.shape[0]

        # for x in range(len(self.out_index)):
        #     nz = np.nonzero(self.out_index[x, :])[0]
        #     print("{} non-zero elements:{}".format(nz.size, nz.tolist()))
        # for x in range(len(self.in_index)):
        #     nz = np.nonzero(self.in_index[x, :])[0]
        #     print("{} non-zero elements:{}".format(nz.size, nz.tolist()))

        # keep numpy arrays; convert to torch tensors in __getitem__

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

        # input and output sizes (excluding batch dimension)
        self.input_size = (self.input_window // self.step_size, self.adjacency.shape[0], len(self.data_channels["source"]))
        self.output_size = (self.pred_window // self.step_size, self.adjacency.shape[0], len(self.data_channels["target"]))

        metadata = {
            "adjacency": np.asarray(self.adjacency, dtype=np.int64),
            "edge_index": np.asarray(self.edge_index, dtype=np.int64),
            "node_coordinates": np.asarray(self.node_coordinates, dtype=np.float32),
            "segment_lengths": np.asarray(self.segment_lengths, dtype=np.float32),
            "cluster_id": np.asarray(self.cluster_id, dtype=np.int64),
            # the grid ID of each road segment
            "grid_id": np.asarray(self.grid_id, dtype=np.int64),
            # the grid (x, y) coordinate of each road segment, shape (N,2)
            "grid_xy": np.asarray(grid_xy, dtype=np.int64),
            # mapping from grid (x, y) coordinate to grid ID
            # converted to string keys for omegaconf compatibility
            "grid_xy_to_id": _tuple_keys_to_str(self.grid_xy_to_id),
            "num_lanes": np.asarray(self.num_lanes, dtype=np.int64),
            "data_stats": {
                "source": self.data_stats["3min"],
                "target": self.data_stats["3min"],
            },
            "data_channels": self.data_channels,
            "data_dim": self.data_dim,
            "data_null_value": self.data_null_value,
            "input_size": self.input_size,  # 3min resolution
            "output_size": self.output_size,  # 3min resolution
        }

        self.metadata = metadata

        return metadata


def initialize_dataset():

    trainset = SimBarcaExplore(
        split="train",
        input_window=30,
        pred_window=30,
        step_size=3,
    )

    testset = SimBarcaExplore(
        split="test",
        input_window=30,
        pred_window=30,
        step_size=3,
    )

    return trainset, testset

def iterate_dataset(trainset, testset):

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
    session_idx = 7
    start_idx = session_idx * trainset.sample_per_session
    for step in range(trainset.sample_per_session):
        new_data = trainset[start_idx + step]
        print(step, new_data["source"].shape, new_data["target"].shape)
        if step >= 5:
            break

def historical_average_baseline(trainset, testset):

    # evaluate trivial average value prediction
    from skymonitor.simbarca_explore_evaluation import SimBarcaExploreEvaluator
    evaluator = SimBarcaExploreEvaluator(
        save_dir="scratch/simbarca_explore_baseline_results",
        ignore_value=0.0
        )
    
    # eval historical average prediction
    density = trainset.veh_density_3min
    flow = trainset.veh_flow_3min

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
    testset_cp.veh_flow_3min = np.asarray(per_location_per_time_avg_flow, dtype=np.float32)
    testset_cp.veh_density_3min = np.asarray(per_location_per_time_avg_density, dtype=np.float32)

    dataloader_cp = torch.utils.data.DataLoader(testset_cp, batch_size=8, shuffle=False, collate_fn=testset_cp.collate_fn)
    _, avg_preds = evaluator.collect_predictions(model=torch.nn.Identity(), data_loader=dataloader_cp, pred_seqs=[], data_seqs=['target'])
    per_location_per_time_avg_prediction = avg_preds['target']  # shape (num_samples, T, P, C)

    per_location_per_time_avg_res = evaluator.calculate_error_metrics(per_location_per_time_avg_prediction, target, data_channels=['flow', 'density'])
    print("Per-Location-Per-Time Average Prediction Results:")
    for key, value in per_location_per_time_avg_res.items():
        print(f"  {key}: {value}")

    # save all results to a json file 
    all_results = {
        "global_average": global_avg_res,
        "per_location_average": per_location_avg_res,
        "per_location_per_time_average": per_location_per_time_avg_res,
    }
    import json
    os.makedirs("scratch/simbarca_explore_baseline_results", exist_ok=True)
    with open("scratch/simbarca_explore_baseline_results/res.json", "w") as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    from skymonitor.visualize import plot_flow_density, visualize_data_as_grid

    trainset, testset = initialize_dataset()
    iterate_dataset(trainset, testset)
    historical_average_baseline(trainset, testset)
    plot_flow_density(trainset.veh_flow_3min, trainset.veh_density_3min, flow_weight=trainset.segment_lengths, note="trainset_flow_density")
    visualize_data_as_grid(trainset.grid_xy, trainset.grid_id, note="grid_id_visualization")