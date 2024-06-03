import os
import json
import tqdm
import pickle
import logging

from pathlib import Path
from collections import defaultdict
from typing import List, Dict
from itertools import product

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style("darkgrid")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import netsanut
from netsanut.data import DATASET_CATALOG


_package_init_file = netsanut.__file__
_root: Path = (Path(_package_init_file).parent.parent).resolve()
assert _root.exists(), "please check detectron2 installation"

logger = logging.getLogger("default")

def add_gaussian_noise(data, std=0.1):
    noise = torch.normal(0, std, size=data.size())
    return data + noise * data

class SimBarca(Dataset):
    data_root = "datasets/simbarca"
    meta_data_folder = "datasets/simbarca/metadata"
    eval_metrics_folder = "datasets/simbarca/eval_metrics"
    input_seqs = ["drone_speed", "ld_speed"]
    output_seqs = ["pred_speed", "pred_speed_regional"]
    train_split_size = 0.75

    def __init__(self, split="train", force_reload=False, use_clean_data=True, filter_short: float = None):
        self.split = split
        self.force_reload = force_reload
        self.read_graph_structure(filter_short=filter_short)
        # data sequences in the dataset 
        self.drone_speed: torch.Tensor
        self.ld_speed: torch.Tensor
        self.pred_speed: torch.Tensor
        self.pred_speed_regional: torch.Tensor
        # metadata for the dataset
        self.adjacency: np.ndarray
        self.edge_index: np.ndarray
        self.node_coordinates: np.ndarray
        self.cluster_id: np.ndarray
        self.grid_id: np.ndarray 
        self.index_to_section_id: Dict[int, int]
        
        samples = self.load_or_process_samples()
        for attribute in self.sequence_names:
            attr_data = torch.as_tensor(samples[attribute], dtype=torch.float32)
            setattr(self, attribute, attr_data)
        self.metadata = self.set_metadata_dict()
        self.data_augmentations = []
        
        if not use_clean_data:
            logger.info("Using corrupted data for train set, but clean label for test set")
            # NOTE quick and dirty data augmentation ... not configuratle, not scalable ... 
            if self.split == "train":
                # we train on corrupted data to make the model more robust
                self.drone_speed = add_gaussian_noise(self.drone_speed, std=0.05)
                self.ld_speed = add_gaussian_noise(self.ld_speed, std=0.15)
                self.pred_speed = add_gaussian_noise(self.pred_speed, std=0.05)
            elif self.split == "test":
                # for test data, the input is corrupted but the label is not for evaulating the model
                self.drone_speed = add_gaussian_noise(self.drone_speed, std=0.05)
                self.ld_speed = add_gaussian_noise(self.ld_speed, std=0.15)
        
        self.edit_graph_structure(filter_short=filter_short)
        
    def __getitem__(self, index):
        # don't use tuple comprehension here, otherwise it will return a generator instead of actual data
        data_seqs = [getattr(self, attribute)[index] for attribute in self.sequence_names]

        return data_seqs

    def __len__(self):
        return self.drone_speed.shape[0]

    @property
    def sequence_names(self):
        return self.input_seqs + self.output_seqs

    def get_processed_file_name(self):
        return "{}/processed/{}.npz".format(self.data_root, self.split)

    def get_split_samples(self, sample_files: List[str]):
        # a sample file contains all training samples from a simulation session
        train_test_split_index = int(self.train_split_size * len(sample_files))

        match self.split:
            case "train":
                return sample_files[:train_test_split_index]
            case "test":
                return sample_files[train_test_split_index:]
            case _:
                raise ValueError("split {} not supported".format(self.split))

    def set_metadata_dict(self):
        metadata = {
            "adjacency": self.adjacency,
            "edge_index": self.edge_index,
            "cluster_id": self.cluster_id,
            "grid_id": self.grid_id,
        }
        
        mean_and_std = dict()
        for att in self.sequence_names:
            seq_data = getattr(self, att)[..., 0]
            seq_data = seq_data[~torch.isnan(seq_data)]
            mean_and_std[att] = (torch.mean(seq_data), torch.std(seq_data))
        metadata["mean_and_std"] = mean_and_std
            
        metadata["input_seqs"] = self.input_seqs
        metadata["output_seqs"] = self.output_seqs

        return metadata

    def read_graph_structure(self, filter_short: float = None):
        """ read the graph structure for the road network from Aimsun-exported metadata.
        """
        folder = "{}/{}".format(_root, self.meta_data_folder)
        connections = pd.read_csv(
            "{}/connections.csv".format(folder),
            dtype={
                "turn": int,
                "org": int,
                "dst": int,
                "intersection": int,
                "length": float,
            },
        )
        link_bboxes = pd.read_csv(
            "{}/link_bboxes_clustered.csv".format(folder),
            dtype={
                "id": int,
                "from_x": float,
                "from_y": float,
                "to_x": float,
                "to_y": float,
                "length": float,
                "out_ang": float,
                "num_lanes": int,
            },
        )
        with open("{}/intersec_polygon.json".format(folder), "r") as f:
            intersection_polygon = json.load(f)

        link_bboxes = link_bboxes.sort_values(by=["id"])
        section_ids_sorted = link_bboxes["id"].to_numpy()
        section_id_to_index = {link_id: index for index, link_id in enumerate(section_ids_sorted)}
        index_to_section_id = {index: section_id for index, section_id in enumerate(section_ids_sorted)}
        section_cluster = link_bboxes["cluster"].to_numpy()  # check with the csv file
        section_grid_id = link_bboxes["grid_nb"].to_numpy()  # check with the csv file
        node_coordinates = link_bboxes[["c_x", "c_y"]].to_numpy()
        segment_lengths = link_bboxes["length"].to_numpy()
        
        adjacency_matrix = np.zeros((len(section_ids_sorted), len(section_ids_sorted)))
        for row in connections.itertuples():
            adjacency_matrix[section_id_to_index[row.org], section_id_to_index[row.dst]] = 1
            # make it symmetric 
            adjacency_matrix[section_id_to_index[row.dst], section_id_to_index[row.org]] = 1
        edge_index = np.array(adjacency_matrix.nonzero())
        
        self.adjacency = adjacency_matrix
        self.segment_lengths = segment_lengths
        self.edge_index = edge_index
        self.node_coordinates = node_coordinates
        self.cluster_id: np.ndarray = section_cluster
        self.grid_id: np.ndarray = section_grid_id
        self.index_to_section_id: Dict = index_to_section_id

    def edit_graph_structure(self, filter_short: float = None):
        # filter out the short sections
        if filter_short is None:
            self.node_filter_mask = None
        else:
            self.adjacency_matrix = delete_nodes(self.adjacency_matrix, self.segment_lengths < filter_short)
            self.edge_index = np.array(self.adjacency_matrix.nonzero())
            self.node_filter_mask = (self.segment_lengths >= filter_short).to_numpy()
            self.node_coordinates = self.node_coordinates[self.node_filter_mask]
            self.section_cluster = self.section_cluster[self.node_filter_mask]
            self.section_grid_id = self.section_grid_id[self.node_filter_mask]
            self.section_ids_sorted = self.section_ids_sorted[self.node_filter_mask]
            self.index_to_section_id = {index: section_id for index, section_id in 
                                        enumerate(self.section_ids_sorted)}
            self.segment_lengths = self.segment_lengths[self.node_filter_mask]
            
        if self.node_filter_mask is not None:
            for attribute in self.sequence_names:
                if attribute == "pred_speed_regional":
                    continue
                # shape (N, T, P, C), apply the node masking on the dimension of P
                setattr(self, attribute, getattr(self, attribute)[:, :, self.node_filter_mask, :])
                
    def load_or_process_samples(self) -> List[Dict[str, pd.DataFrame | np.ndarray]]:
        if Path(self.get_processed_file_name()).exists() and not self.force_reload:
            print("Trying to load existing processed samples for {} split".format(self.split))
            with open(self.get_processed_file_name(), "rb") as f:
                loaded_data = np.load(f)
                return {key: value for key, value in loaded_data.items()}

        print("No processed samples found or forced to reload, processing samples from scratch")
        all_sample_data = defaultdict(list)
        sample_files = sorted(
            list(Path("{}/{}".format(_root, self.data_root)).glob("simulation_sessions/session_*/timeseries/samples.npz"))
        )
        split_sample_files = self.get_split_samples(sample_files)

        print("Found {} samples for {} split, reading them one by one".format(len(split_sample_files), self.split))
        for sample_file in tqdm.tqdm(split_sample_files):
            with open(sample_file, "rb") as f:
                sample_data: np.lib.npyio.NpzFile = np.load(f)
                for key in sample_data:
                    all_sample_data[key].append(sample_data[key])

        # concatenate the samples along the batch dimension
        for key, value in all_sample_data.items():
            all_sample_data[key] = np.concatenate(value, axis=0)

        print("Processing per section data")
        # process the input and predicted speed
        processed_samples = dict()
        
        # loop detector readings, no extra computations needed after stacking
        processed_samples["ld_speed"] = all_sample_data["ld_speed"]
        processed_samples["pred_ld_speed"] = all_sample_data["pred_ld_speed"]
        
        # input and predict speed per section, using v = total_dist / total_time
        # each array in all_sample_data have shape (N, T, P, 2), where N is the number of samples, T is the number of time steps, P is the number of sections, and the last dimension 2 corresponds to (time_in_day, value)
        for mod_type in ["drone", "pred"]:
            vdist_values: np.ndarray = all_sample_data["{}_vdist".format(mod_type)][..., 0] # total vehicle distance
            vdist_tind: np.ndarray = all_sample_data["{}_vdist".format(mod_type)][..., 1] # time in day
            vtime_values: np.ndarray = all_sample_data["{}_vtime".format(mod_type)][..., 0] # total vehicle time
            speed_values = vdist_values / vtime_values
            # leave the invalid number as nan, and let the model decide how to deal with the nan values
            # because the nan values here it means both distance and time are 0, no vehicle detected
            # this is different from zero speed, which can also happen when all vehicles are stopped at red lights
            # speed_values = np.nan_to_num(speed_values, nan=-1)
            processed_samples["{}_speed".format(mod_type)] = np.stack([speed_values, vdist_tind], axis=-1)
        
        print("Processing regional data")
        regional_speed = []
        # aggregate link into regions
        for region_id in np.unique(self.cluster_id):
            region_mask = self.cluster_id == region_id
            # sum the total distance but ignore NaN values, `np.sum` will be NaN if one element is NaN
            region_vdist_values = np.nansum(all_sample_data["pred_vdist"][..., region_mask, 0], axis=-1)
            # add the time in day here, the first index
            # time in day was copied for all positions, so taking 1 is enough
            region_vdist_tind = all_sample_data["pred_vdist"][..., region_mask, 1][..., 0]
            region_vtime_values = np.nansum(all_sample_data["pred_vtime"][..., region_mask, 0], axis=-1)
            region_speed_values = region_vdist_values / region_vtime_values
            # the regional speed is very unlikely to be nan, but we still don't exclude this possibility
            # region_speed_values = np.nan_to_num(region_speed_values, nan=-1)
            regional_speed.append(np.stack([region_speed_values, region_vdist_tind], axis=-1))
        # the elements have shape (N, T, 2), where 2 corresponds to (time_in_day, value)
        # we stack them into shape (N, T, R, 2) where R is the number of regions
        processed_samples["pred_speed_regional"] = np.stack(regional_speed, axis=2)

        # we also keep the vehicle distances and time for the drone
        processed_samples["drone_vdist"] = all_sample_data["drone_vdist"]
        processed_samples["pred_vdist"] = all_sample_data["pred_vdist"]
        processed_samples["drone_vtime"] = all_sample_data["drone_vtime"]
        processed_samples["pred_vtime"] = all_sample_data["pred_vtime"]

        # save all samples as a compressed npz file
        print("Saving processed samples to {}".format(self.get_processed_file_name()))
        with open(self.get_processed_file_name(), "wb") as f:
            np.savez_compressed(f, **processed_samples)

        return processed_samples

    def collate_fn(self, list_of_seq: List[Dict]) -> Dict[str, torch.Tensor]:
        data_dict = dict()
        for attr_id, attribute in enumerate(self.sequence_names):
            data_dict[attribute] = torch.cat(
                [seq[attr_id].unsqueeze(0) for seq in list_of_seq], dim=0
            ).contiguous()

        # assume the output values have 0 index in the last dimension,
        # the other dimensions are time in day, day in week etc.
        for name in self.output_seqs:
            data_dict[name] = data_dict[name][..., 0]

        for aug in self.data_augmentations:
            data_dict = aug(data_dict)

        data_dict["metadata"] = self.metadata

        return data_dict
    
    @staticmethod
    def visualize_batch(data_dict, pred_dict=None, save_dir="./", batch_num=0, section_num=573, save_note="example"):
        
        # plot input and output 
        b, s = batch_num, section_num
        cluster_id = data_dict['metadata']['cluster_id']
        drone_in = data_dict['drone_speed'].cpu().numpy()
        ld_in = data_dict['ld_speed'].cpu().numpy()

        label = data_dict['pred_speed'].cpu().numpy()
        label_regional = data_dict['pred_speed_regional'].cpu().numpy()
        in1 = drone_in[b, :, s, 0]
        tin1 = np.linspace(0, 30, len(in1))
        in2 = ld_in[b, :, s, 0]
        tin2 = np.linspace(0, 30, len(in2))
        label1 = label[b, :, s]
        tlabel1 = np.linspace(33, 60, len(label1))
        label2 = label_regional[b, :, cluster_id[s]]
        tlabel2 = np.linspace(33, 60, len(label2))
        
        # draw the model predictions if available
        if pred_dict is not None:
            pred = pred_dict['pred_speed'].cpu().numpy()
            pred_regional = pred_dict['pred_speed_regional'].cpu().numpy()
            out1 = pred[b, :, s]
            tout1 = np.linspace(33, 60, len(out1))
            out2 = pred_regional[b, :, cluster_id[s]]
            tout2 = np.linspace(33, 60, len(out2))
            
        fig, ax = plt.subplots(figsize=(6.5, 4))

        ax.plot(tin1, in1, label="drone_input")
        ax.plot(tin2, in2, label="ld_input")
        ax.plot(tlabel1, label1, label="label_segment")
        ax.plot(tlabel2, label2, label="label_regional")
        
        try:
            ax.plot(tout1, out1, label="pred_segment")
            ax.plot(tout2, out2, label="pred_regional")
        except:
            pass
        
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Speed (m/s)")
        ax.legend()
        fig.tight_layout()
        fig.savefig("{}/pred_sample_b{}s{}_{}.pdf".format(save_dir, b, s, save_note))

    @staticmethod
    def plot_pred_for_section(all_preds, all_labels, save_dir="./", section_num=100, save_note="example"):

        p = section_num
        
        y1 = all_preds['pred_speed'][:, -1, p]
        y2 = all_labels['pred_speed'][:, -1, p]
        xx = np.arange(len(y1))

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(xx, y1, label='30min_pred')
        ax.plot(xx, y2, label='GT', alpha=0.5)
        # add vertical line every 20 time steps
        for i in range(0, len(xx), 20):
            ax.axvline(i, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.legend()
        ax.set_xlabel("Time step (not exactly ...)")
        ax.set_ylabel("Speed (m/s)")
        
        fig.tight_layout()
        fig.savefig("{}/30min_ahead_pred_{}_{}.pdf".format(save_dir, p, save_note))

    @staticmethod
    def plot_MAE_by_location(node_coordinates, all_preds, all_labels, save_dir="./", save_note="example"):
        MAE = torch.abs(all_preds['pred_speed'] - all_labels['pred_speed'])
        mae_by_section = torch.nanmean(MAE, dim=(0,1))
        fig, ax = plt.subplots(figsize=(6.5, 4))
        im = ax.scatter(node_coordinates[:, 0], node_coordinates[:, 1], c=mae_by_section.numpy(), s=2)
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("X Coordinates")
        ax.set_ylabel("Y Coordinates")
        fig.tight_layout()
        fig.savefig("{}/average_mae_{}.pdf".format(save_dir, save_note))
        
def delete_nodes(adj_mtx, delete_mask):
    # get the indexes of the nodes to be deleted
    deleted_nodes = np.nonzero(delete_mask)[0]

    # For each deleted node, connect its neighbors directly
    for node in deleted_nodes:
        in_nodes = np.nonzero(adj_mtx[:, node])[0]
        out_nodes = np.nonzero(adj_mtx[node, :])[0]
        for i, j in product(in_nodes, out_nodes):
            if i == j: # avoid self-loop
                continue
            adj_mtx[i, j] = 1

    # remove rows and columns of deleted nodes
    adj_mtx = np.delete(adj_mtx, deleted_nodes, axis=0)
    adj_mtx = np.delete(adj_mtx, deleted_nodes, axis=1)

    return adj_mtx

class SimBarcaRandomObservation(SimBarca):
    """ This class implements randomized drone observations and loop detector observations. The purpose is to make the dataset more realistic, because in reality, we don't have loop detectors for all road segments, and we can hardly fly enough drones to cover a whole city. Therefore, dealing with partial information is inevitable. 

    Concretely, 10% of the road segments will have loop detector observations, which will be initialized at the first time and later saved to file for reuse. In this way the loop detector positions are fixed across experiments, and the results can be fairly compared. 
    
    The drone observaions will be available for random 10% of the grid IDs (but not directly road segments), since we assume a drone to observe nearly all vehicles in its square-shaped FOV. To imitate a real-world scenario, each simulation session will be regarded as an individual day, when we fly the drones in different ways. We want the drone positions to mimic the flight plan for the simulation sessions, which will be consistent in all epochs in the same experiment and also across all experiments.
    
    While it's simple that the training and testing splits should share the same loop detector positions, the drone positions are more complicated. The training samples are generated in a sliding-window way based on the traffic statistics of the whole simulation. Therefore the drone positions should also be a "sliding-window", which means neighboring samples should have overlapped drone positions with only 1 time step difference. Check `load_or_init_drone_pos` for implementation. 
    
    Besides, for the training split, we exclude the data of the unmonitored road segments from both the input and output, which means partial input and partial label for the model to learn. The labels for regional speed predictions will also be created based on partial information (which makes it biased but that's the best we could do for now). However, for the test set, we still use the full information for evaluation (so don't train on the test set).
    """
    
    def __init__(self, ld_per=0.1, drone_per=0.1, reinit_pos = False, random_state = 42, **kwargs):
        super().__init__(**kwargs)
        self.sample_per_session = 20 # hard coded for now ... better saved in sample files
        self.ld_per = ld_per
        self.drone_per = drone_per
        self.random_state = random_state
        self.reinit_pos = reinit_pos
        self.ld_mask: torch.Tensor
        self.drone_mask: torch.Tensor
        self.load_or_init_ld_mask()
        self.load_or_init_drone_mask()
    
    @property
    def ld_mask_file(self):
        return "{}/processed/ld_mask.pkl".format(self.data_root)

    @property
    def drone_mask_file(self):
        return "{}/processed/drone_mask_{}.pkl".format(self.data_root, self.split)
    
    def load_or_init_ld_mask(self):
        if os.path.exists(self.ld_mask_file) and not self.reinit_pos:
            with open(self.ld_mask_file, "rb") as f:
                ld_mask = pickle.load(f)
            if ld_mask.sum() != int(self.ld_per * self.adjacency.shape[0]):
                logger.warning("The loop detector coverage in the file are not consistent with the current settings, check `ld_per` argument or set `reinit_pos=True`")
        else:
            rng = np.random.default_rng(self.random_state)
            # randomly select a few indexes of the road segments to have loop detectors
            ld_pos = rng.choice(
                    np.arange(self.adjacency.shape[0]), 
                    size=int(self.ld_per * self.adjacency.shape[0]), 
                    replace=False
                    )
            ld_mask = np.zeros(shape=self.adjacency.shape[0], dtype=bool)
            ld_mask[ld_pos] = True
            
            with open(self.ld_mask_file, "wb") as f:
                pickle.dump(ld_mask, f)
        
        self.ld_mask = torch.as_tensor(ld_mask, dtype=torch.long)

    def load_or_init_drone_mask(self):
        
        if os.path.exists(self.drone_mask_file) and not self.reinit_pos:
            all_drone_mask = pickle.load(open(self.drone_mask_file, "rb"))
            if all_drone_mask.shape[-1] != int(self.drone_per * len(self.grid_id)):
                logger.warning("The drone coverage in the file are not consistent with the current settings, check `drone_per` argument or set `reinit_pos=True`")
        else:
            grid_cells = np.sort(np.unique(self.grid_id))
            rng = np.random.default_rng(self.random_state + 777) # avoid using the same random seed as ld_pos
            
            all_drone_mask = []
            for _ in range(len(self) // self.sample_per_session): # different simulation session
                # init drone positions for the first sample in each session
                # 30min input, 30min output, change every 3 mins, so that's 20 x num_grid 
                drone_mask = np.stack(
                    [np.isin(self.grid_id, 
                             rng.choice(grid_cells, size=int(self.drone_per * len(grid_cells)), replace=False)
                    ) for _ in range(20)]
                )
                all_drone_mask.append(drone_mask)
                # for every 3 min, sample a new set of drone positions and discard the earliest step
                for _ in range(1, self.sample_per_session):
                    next_step_drone_mask = np.isin(
                        self.grid_id, 
                        rng.choice(grid_cells, size=(int(self.drone_per * len(grid_cells))), replace=False)
                    )
                    drone_mask = np.concatenate((drone_mask[1:, :], next_step_drone_mask.reshape(1, -1)), axis=0)
                    all_drone_mask.append(drone_mask)
            all_drone_mask = np.stack(all_drone_mask, axis=0)
            with open(self.drone_mask_file, "wb") as f:
                pickle.dump(all_drone_mask, f)
        
        self.drone_mask = torch.as_tensor(all_drone_mask, dtype=torch.long)
        
    def apply_masking(self, dict):
        """ Apply the masking of loop detector and drone data to the input and label
        
            For both train and test sets:
                1. Set all INPUT modalities for unmonitored road segments to nan
            For train set only:
                1. Set all OUTPUT modalities for unmonitored road segments to nan
                2. Recalculate regional speed values based on the monitored road segments

            For the test set, we keep the output values as they are, because we want to evaluate the model's performance on the full information, even when the model is trained with partial data.
        """
        pass
    
    def __getitem__(self, index):
        """ 
        In the files, we save the drone mask every 3 mins, since we assume the drones will move to monitor different locations every 3 minutes. However, the drone input is given every 5 seconds, so we need to extend the drone mask to the same time resolution as the drone input.
        
        The mask for loop detector is a 0-1 tensor of shape P, where P is the number of road segments. Since loop detectors are installed as fixed infrastructure, ld_mask remain unchanged over time.
        The mask for drone input of ONE sample is also a 0-1 tensor but it has shape (T, P), where P is the number of road segments, and T is the number of time steps. To avoid complex transition states, we assume drones to jump from one grid to another. 
        """
        data_dict = super().__getitem__(index)
        data_dict['ld_mask'] = self.ld_mask
        data_dict['drone_mask'] = self.drone_mask[index]
        
        return self.apply_masking(data_dict)
        
    def collate_fn(self, list_of_seq: List[Dict]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("This method is not implemented yet")
    
    def visualize_drone_pos(self):
        pass
    
        
if __name__.endswith(".simbarca"):
    """this happens when something is imported from this file
    we can register the dataset here
    """
    DATASET_CATALOG['simbarca_train'] = lambda **args: SimBarca(split='train', **args)
    DATASET_CATALOG['simbarca_test'] = lambda **args: SimBarca(split='test', **args)
    DATASET_CATALOG['simbarca_rnd_train'] = lambda **args: SimBarcaRandomObservation(split='train', **args)
    DATASET_CATALOG['simbarca_rnd_test'] = lambda **args: SimBarcaRandomObservation(split='test', **args)

if __name__ == "__main__":
    
    from netsanut.utils.event_logger import setup_logger
    logger = setup_logger(name="default", level=logging.INFO)
    
    train_set = SimBarca(split="train", force_reload=True, filter_short=10)
    test_set = SimBarca(split="test", force_reload=True, filter_short=10)
    
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=train_set.collate_fn)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False, collate_fn=test_set.collate_fn)

    for data_dict in train_loader:
        SimBarca.visualize_batch(data_dict)
        break

    for data_dict in test_loader:
        SimBarca.visualize_batch(data_dict)
        break
    
    debug_set = SimBarcaRandomObservation(split='train', random_state=42)
    sample = debug_set[0]
