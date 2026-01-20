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
        - Low-frequency 3-minute aggregates for observation and prediction.
        - High-frequency 5-second aggregates are available but not used for input.

    The parent class `SimBarcaForecast` loads the flow, density for all sessions as a tensor bundle shaped `[num_sessions, time, num_locations]`.
    Each session behaves like an RL environment rollout, and we want to train our drone policy using all sessions in the training set, and will test it on all sessions in the test set.

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
            augmentations" : a list of data augmentation modules to apply during data loading, for supervised training only.
        """
        # the number of sliding window samples during the 120 minute valid simulation
        num_unpadded_samples = ( 120 - (input_window + pred_window) ) // step_size
        super().__init__(split, input_window, pred_window, step_size, num_unpadded_samples)
        self._active_session = None  # Current session data
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

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """ Get the idx-th pair of (past, future) data sample from the active session.
            works for both supervised training and RL data retrieval (through iterate_active_session).
        """

        active_session = self.active_session if self.active_session is not None else (idx // self.sample_per_session)
        sample_idx = idx % self.sample_per_session
        if (self.active_session is not None) and (idx >= self.sample_per_session):
            print("WARNING: idx >= sample_per_session when active_session is set.")
    
        sample_in_index = self.in_index[sample_idx]
        sample_out_index = self.out_index[sample_idx]

        time_in_day_3min = repeat(
            self.time_in_day_3min[self.in_index[sample_idx]],
            "t -> t n",
            n=self.veh_flow_3min.shape[-1],
        )

        # stack channels along the last dim
        # past: 3-min resolution [Tin, N, 4] => (flow, density, speed, time_in_day), without speed, the feature dimension becomes 3
        source = torch.stack(
            [
                self.veh_flow_3min[active_session, sample_in_index, :],
                self.veh_density_3min[active_session, sample_in_index, :],
                # self.veh_speed_3min[active_session, sample_in_index, :],
                time_in_day_3min,
            ],
            dim=-1,
        )

        # future: low-frequency resolution [Tout, N, 3] => (flow, density, speed)
        target = torch.stack(
            [
                self.veh_flow_3min[active_session, sample_out_index, :],
                self.veh_density_3min[active_session, sample_out_index, :],
                # self.veh_speed_3min[active_session, sample_out_index, :],
            ],
            dim=-1,
        )

        return {"source": source, "target": target}

    def collate_fn(self, list_of_data_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
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
        try:
            idx = int(idx)
        except Exception as e:
            raise ValueError(f"Active session should be an integer: {e}")
        self._active_session = idx % self.num_sessions  # this also works for interger tensors

    def iterate_active_session(self):
        """Iterate within the current active session."""
        if self._active_session is None:
            raise ValueError("Active session not set.")
        for i in range(self.sample_per_session):
            yield i, self.__getitem__(i)

    def prepare_data_sequences(self):
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
        self.input_size = (self.input_window // self.step_size, self.adjacency.shape[0], len(self.data_channels["source"]))
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


def initialize_dataset():

    trainset = SimBarcaExplore(
        split="train",
        input_window=30,
        pred_window=30,
        step_size=3,
    )
    # trainset.summarize()

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
    trainset.active_session = 7  # set the active session
    for step, new_data in trainset.iterate_active_session():
        print(step, new_data["source"].shape, new_data["target"].shape)
        if step >= 5:
            break

def historical_average_baseline(trainset, testset):

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

def compute_critical_density(trainset, visualize=True) -> np.ndarray:
    """
    Compute per-road-segment critical density using the 85th percentile over all training sessions.

    Args:
        trainset: Dataset containing 3-minute density arrays shaped [num_sessions, time, num_locations].
        visualize: If True, also save a map of critical density to the scratch folder.

    Returns:
        np.ndarray: Critical density per road segment with shape (num_locations,).
    """
    densities = trainset.veh_density_3min
    density_array = np.asarray(densities, dtype=np.float32)
    flattened = density_array.reshape(-1, density_array.shape[-1])
    critical_density = np.quantile(flattened, 0.85, axis=0)
    if visualize:
        map_visualize(
            node_coordinate=trainset.node_coordinates,
            values=critical_density,
            figure_note=f"Critical density (85th percentile) [{trainset.split}]",
            save_path=os.path.join("scratch", f"critical_density_map_{trainset.split}.png"),
            cmap="coolwarm",
            colorbar_label="Critical density (veh/m)",
        )
    return critical_density

def visualize_congestion_front(dataset, critical_density):
    """
    Visualize and animate the ratio of density to critical density for each simulation session.

    For every session in `dataset`, this function creates a scatter plot of all road segments where
    colors encode the density-to-critical-density ratio at each time step, then saves the animation
    under `./scratch/congestion_front_<split>/session_XXX.gif`.

    Args:
        dataset: Dataset providing `veh_density_3min`, `node_coordinates`, `num_sessions`, and `session_ids`.
        critical_density: Per-road-segment critical density tensor or array with shape (num_locations,).
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation

    output_dir = os.path.join("scratch", f"congestion_front_{getattr(dataset, 'split', 'dataset')}")
    os.makedirs(output_dir, exist_ok=True)

    densities = dataset.veh_density_3min
    density_array = np.asarray(densities, dtype=np.float32)
    critical_array = np.asarray(critical_density, dtype=np.float32)
    critical_array = np.maximum(critical_array, 1e-6)

    ratios = density_array / critical_array
    ratios = np.clip(ratios, a_min=None, a_max=2.0)
    coords = np.asarray(dataset.node_coordinates, dtype=np.float32)
    session_simids = dataset.session_ids
    demand_scales = dataset.demand_scales

    for idx in range(dataset.num_sessions):
        print(f"Visualizing session {session_simids[idx]} ({idx+1}/{dataset.num_sessions})...")
        session_ratios = ratios[idx]
        scale = demand_scales[idx]
        title_prefix = f"Session {session_simids[idx]} (scale={scale:.2f})"

        fig, ax, scatter = map_visualize(
            node_coordinate=coords,
            values=session_ratios[0],
            figure_note=f"{title_prefix} density ratio (t=0)",
            save_path=None,
            cmap="coolwarm",
            vmin=0.0,
            vmax=2.0,
            colorbar_label="Density / Critical",
        )

        def update(frame):
            scatter.set_array(session_ratios[frame])
            ax.set_title(f"{title_prefix} density ratio (t={frame})")
            return scatter,

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=session_ratios.shape[0],
            interval=300,
            blit=False,
        )

        save_path = os.path.join(output_dir, f"session_{session_simids[idx]:03d}.gif")
        anim.save(save_path, writer=animation.PillowWriter(fps=4))
        plt.close(fig)

def map_visualize(
    node_coordinate: np.ndarray,
    values: np.ndarray,
    figure_note: str,
    save_path: str = None,
    cmap: str = "coolwarm",
    vmin: float = None,
    vmax: float = None,
    colorbar_label: str = "Value",
):
    """
    Scatter map helper used for static plots and animation frames.

    Args:
        node_coordinate: Array of shape (num_locations, 2) with XY coordinates.
        values: Array of shape (num_locations,) containing values to visualize.
        figure_note: Title text for the figure.
        save_path: Optional file path to save the figure.
        cmap: Matplotlib colormap name.
        vmin: Optional minimum for color scaling.
        vmax: Optional maximum for color scaling.
        colorbar_label: Label for the colorbar.

    Returns:
        Tuple of (figure, axis, scatter) for reuse (e.g., animation updates).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="darkgrid")

    coords = np.asarray(node_coordinate, dtype=np.float32)
    values = values.numpy() if isinstance(values, torch.Tensor) else np.asarray(values, dtype=np.float32)

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap=cmap, vmin=vmin, vmax=vmax, s=8)
    fig.colorbar(scatter, ax=ax, label=colorbar_label)
    ax.set_title(figure_note)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_aspect("equal")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig, ax, scatter

if __name__ == "__main__":

    trainset, testset = initialize_dataset()
    # iterate_dataset(trainset, testset)
    # historical_average_baseline(trainset, testset)
    critical_density = compute_critical_density(trainset)
    visualize_congestion_front(trainset, critical_density)
    visualize_congestion_front(testset, critical_density)
