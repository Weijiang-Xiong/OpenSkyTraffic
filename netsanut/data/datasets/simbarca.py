import os
import json
import tqdm
import pickle
import logging
import argparse

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
assert _root.exists(), "please check package installation"

logger = logging.getLogger("default")

def add_gaussian_noise(generator:torch.Generator, data:torch.Tensor, std=0.1):
    noise = torch.normal(0, std, size=data.size()[:-1], generator=generator)
    data[:, :, :, 0] = data[:, :, :, 0] + noise * data[:, :, :, 0]
    return data

def session_number_from_path(path):
    import re
    return int(re.search(r"session_(\d+)", path).group(1))

class SimBarca(Dataset):
    data_root = "datasets/simbarca"
    meta_data_folder = "{}/metadata".format(data_root)
    eval_metrics_folder = "{}/eval_metrics".format(data_root)
    session_splits = "{}/train_test_split.json".format(meta_data_folder)
    session_folder_pattern = "simulation_sessions/session_*"
    soi_file = "{}/sections_of_interest.txt".format(meta_data_folder)
    
    input_seqs = ["drone_speed", "ld_speed"] # input sequences to feed to the model
    output_seqs = ["pred_speed", "pred_speed_regional"] # output sequences to predict
    add_seqs = [] # additional sequences that are required in the input batch, not time series of traffic variables (like the monitoring mask in SimbarcaRandomObservation)
    aux_seqs = [] # sequences that are required during data processing, they are traffic variables but are not input or output, needed in collate but deleted afterwards (vkt and vht for recomputing regional label)
    
    def __init__(self, split="train", force_reload=False, use_clean_data=True, filter_short: float = None, noise_seed=114514):
        self.sample_per_session = 20 # hard coded for now ... better saved in sample files
        self.split = split
        self.force_reload = force_reload
        self.read_graph_structure()
        # data sequences in the dataset 
        self.drone_speed: torch.Tensor
        self.ld_speed: torch.Tensor
        self.pred_speed: torch.Tensor
        self.pred_speed_regional: torch.Tensor
        # metadata for the dataset
        self.adjacency: torch.Tensor
        self.edge_index: torch.Tensor
        self.node_coordinates: torch.Tensor
        self.cluster_id: torch.Tensor
        self.grid_id: torch.Tensor 
        self.index_to_section_id: Dict[int, int]
        self.session_ids: torch.Tensor # shape (N, ) where N is the number of samples
        self.demand_scales: torch.Tensor # shape (N, ) where N is the number of samples
        self.noise_seed = noise_seed
        
        samples = self.load_or_process_samples()
        for attribute in self.io_seqs + self.aux_seqs:
            attr_data = torch.as_tensor(samples[attribute], dtype=torch.float32)
            setattr(self, attribute, attr_data)
            del samples[attribute] # free memory otherwise 64 GB won't be enough ... 
        self.metadata = self.set_metadata_dict()
        self.session_ids, self.demand_scales = self.get_session_properties(fit_dataset_len=True)
        
        self.data_augmentations = []
        
        if not use_clean_data:
            
            logger.info("Using random seed {} for adding noise to the data".format(self.noise_seed))
            self.rnd_generator = torch.Generator()
            self.rnd_generator.manual_seed(self.noise_seed)
            
            logger.info("Using corrupted data for train set, but clean label for test set")
            # NOTE quick and dirty data augmentation ... not configuratle, not scalable ... 
            if self.split == "train":
                logger.info("Adding Gaussian noise to the training input and label")
                # we train on corrupted data to make the model more robust
                self.drone_speed = add_gaussian_noise(self.rnd_generator, self.drone_speed, std=0.05)
                self.ld_speed = add_gaussian_noise(self.rnd_generator, self.ld_speed, std=0.15)
                self.pred_speed = add_gaussian_noise(self.rnd_generator, self.pred_speed, std=0.05)
                self.pred_speed_regional = add_gaussian_noise(self.rnd_generator, self.pred_speed_regional, std=0.05)
            elif self.split == "test":
                # for test data, the input is corrupted but the label is not for evaulating the model
                logger.info("Adding Gaussian noise to the testing input (BUT NOT lable)")
                self.drone_speed = add_gaussian_noise(self.rnd_generator, self.drone_speed, std=0.05)
                self.ld_speed = add_gaussian_noise(self.rnd_generator, self.ld_speed, std=0.15)
        
        # self.edit_graph_structure(filter_short=filter_short)
        
        # load the sections of interest to visualize segment-level predictions
        with open(self.soi_file, "r") as f:
            self.sections_of_interest = [int(x) for x in f.read().split(",")]
            
    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        # don't use tuple comprehension here, otherwise it will return a generator instead of actual data
        data_seqs = {attr: getattr(self, attr)[index] for attr in self.io_seqs + self.aux_seqs}

        return data_seqs

    def __len__(self):
        return self.drone_speed.shape[0]

    @property
    def io_seqs(self):
        """ all the traffic variable series that need to be returned in a batch
        """
        return self.input_seqs + self.output_seqs
    
    @property
    def processed_file(self):
        return "{}/processed/{}.npz".format(self.data_root, self.split)

    def get_sessions_in_split(self) -> List[str]:
        """ return a list of paths that contains the simulation sessions in the split
        """
        
        if not os.path.exists(self.session_splits):
            print("No train_test_split.json file found, please use `scripts/data/simbarca/choose_train_test.py`")
        with open(self.session_splits, "r") as f:
            session_ids = json.load(f)[self.split]
            
        session_folders = sorted(
            list(Path("{}/{}".format(_root, self.data_root)).glob(self.session_folder_pattern))
        )
        sessions_in_split =[str(f) for f in session_folders if session_number_from_path(str(f)) in session_ids]
        
        return sessions_in_split

    def set_metadata_dict(self):
        metadata = {
            "adjacency": self.adjacency,
            "edge_index": self.edge_index,
            "cluster_id": self.cluster_id,
            "grid_id": self.grid_id,
        }
        
        mean_and_std = dict()
        for att in self.io_seqs:
            seq_data = getattr(self, att)[..., 0]
            seq_data = seq_data[~torch.isnan(seq_data)]
            mean_and_std[att] = (torch.mean(seq_data), torch.std(seq_data))
        metadata["mean_and_std"] = mean_and_std
            
        metadata["input_seqs"] = self.input_seqs
        metadata["output_seqs"] = self.output_seqs

        return metadata

    def get_session_properties(self, fit_dataset_len=True):
        
        sessions_in_split = self.get_sessions_in_split()
            
        session_ids, demand_scales = [], []
        for f in sessions_in_split:
            scale = json.load(open("{}/settings.json".format(f), 'r'))["global_scale"]
            session_id = session_number_from_path(f)
            session_ids.append(session_id)
            demand_scales.append(scale)
        
        if fit_dataset_len:
            # repeat the session ids and demand scales to match the number of samples
            session_ids = np.repeat(session_ids, self.sample_per_session)
            demand_scales = np.repeat(demand_scales, self.sample_per_session)
        
        return torch.as_tensor(session_ids), torch.as_tensor(demand_scales)
        
    def read_graph_structure(self):
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
        cluster_id = link_bboxes["cluster"].to_numpy()  # check with the csv file
        section_grid_id = link_bboxes["grid_nb"].to_numpy()  # check with the csv file
        node_coordinates = link_bboxes[["c_x", "c_y"]].to_numpy()
        segment_lengths = link_bboxes["length"].to_numpy()
        
        adjacency_matrix = np.zeros((len(section_ids_sorted), len(section_ids_sorted)))
        for row in connections.itertuples():
            adjacency_matrix[section_id_to_index[row.org], section_id_to_index[row.dst]] = 1
            # make it symmetric 
            adjacency_matrix[section_id_to_index[row.dst], section_id_to_index[row.org]] = 1
        edge_index = np.array(adjacency_matrix.nonzero())
        
        self.adjacency: torch.Tensor = torch.as_tensor(adjacency_matrix)
        self.segment_lengths: torch.Tensor = torch.as_tensor(segment_lengths)
        self.edge_index: torch.Tensor = torch.as_tensor(edge_index)
        self.node_coordinates: torch.Tensor = torch.as_tensor(node_coordinates)
        self.cluster_id: torch.Tensor = torch.as_tensor(cluster_id)
        self.grid_id: torch.Tensor = torch.as_tensor(section_grid_id)
        # self.section_ids_sorted: torch.Tensor = torch.as_tensor(section_ids_sorted)
        self.index_to_section_id: Dict = index_to_section_id

    # def edit_graph_structure(self, filter_short: float = None):
    #     # filter out the short sections
    #     if filter_short is None:
    #         self.node_filter_mask = None
    #     else:
    #         self.adjacency = delete_nodes(self.adjacency, self.segment_lengths < filter_short)
    #         self.edge_index = self.adjacency.nonzero()
    #         self.node_filter_mask = (self.segment_lengths >= filter_short)
    #         self.node_coordinates = self.node_coordinates[self.node_filter_mask]
    #         self.cluster_id = self.cluster_id[self.node_filter_mask]
    #         self.grid_id = self.grid_id[self.node_filter_mask]
    #         self.section_ids_sorted = self.section_ids_sorted[self.node_filter_mask]
    #         self.index_to_section_id = {index: section_id for index, section_id in 
    #                                     enumerate(self.section_ids_sorted)}
    #         self.segment_lengths = self.segment_lengths[self.node_filter_mask]
            
    #     if self.node_filter_mask is not None:
    #         for attribute in self.io_seqs:
    #             if attribute == "pred_speed_regional":
    #                 continue
    #             # shape (N, T, P, C), apply the node masking on the dimension of P
    #             setattr(self, attribute, getattr(self, attribute)[:, :, self.node_filter_mask, :])
    
    def regional_speed_from_segment(self, seg_vdist, seg_vtime) -> np.ndarray:
        """ compute regional speed from segment level vehicle travel distance and travel time
        """
        regional_speed = []
        # aggregate link into regions
        for region_id in np.unique(self.cluster_id):
            region_mask = self.cluster_id == region_id
            # sum the total distance but ignore NaN values, `np.sum` will be NaN if one element is NaN
            region_vdist_values = np.nansum(seg_vdist[..., region_mask, 0], axis=-1)
            # add the time in day here, the first index
            # time in day was copied for all positions, so taking 1 is enough
            region_vdist_tind = seg_vdist[..., region_mask, 1][..., 0]
            region_vtime_values = np.nansum(seg_vtime[..., region_mask, 0], axis=-1)
            region_speed_values = region_vdist_values / region_vtime_values
            # the regional speed is very unlikely to be nan, but we still don't exclude this possibility
            # region_speed_values = np.nan_to_num(region_speed_values, nan=-1)
            regional_speed.append(np.stack([region_speed_values, region_vdist_tind], axis=-1))
        # the elements have shape (N, T, 2), where 2 corresponds to (time_in_day, value)
        # we stack them into shape (N, T, R, 2) where R is the number of regions
        regional_speed_array = np.stack(regional_speed, axis=2)
        return regional_speed_array
    
    def load_or_process_samples(self) -> List[Dict[str, pd.DataFrame | np.ndarray]]:
        if Path(self.processed_file).exists() and not self.force_reload:
            print("Trying to load existing processed samples for {} split".format(self.split))
            with open(self.processed_file, "rb") as f:
                loaded_data = np.load(f)
                return {key: value for key, value in loaded_data.items() if key in self.io_seqs + self.aux_seqs}

        print("No processed samples found or forced to reload, processing samples from scratch")
        all_sample_data = defaultdict(list)
        split_sample_files =["{}/timeseries/samples.npz".format(f) for f in self.get_sessions_in_split()]

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
        processed_samples["pred_speed_regional"] = self.regional_speed_from_segment(
            all_sample_data["pred_vdist"], all_sample_data["pred_vtime"]
        )

        # we also keep the vehicle distances and time for the drone
        processed_samples["drone_vdist"] = all_sample_data["drone_vdist"]
        processed_samples["pred_vdist"] = all_sample_data["pred_vdist"]
        processed_samples["drone_vtime"] = all_sample_data["drone_vtime"]
        processed_samples["pred_vtime"] = all_sample_data["pred_vtime"]

        # save all samples as a compressed npz file
        print("Saving processed samples to {}".format(self.processed_file))
        with open(self.processed_file, "wb") as f:
            np.savez_compressed(f, **processed_samples)
        
        return processed_samples

    def collate_fn(self, list_of_seq: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_data = dict()
        for attr in self.io_seqs + self.add_seqs:
            batch_data[attr] = torch.cat(
                [seq[attr].unsqueeze(0) for seq in list_of_seq], dim=0
            ).contiguous()

        # assume the output values have 0 index in the last dimension,
        # the other dimensions are time in day, day in week etc.
        for name in self.output_seqs:
            batch_data[name] = batch_data[name][..., 0]

        for aug in self.data_augmentations:
            batch_data = aug(batch_data)

        batch_data["metadata"] = self.metadata

        return batch_data
    
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
        plt.close()
        logger.info("Saved the plot to {}/pred_sample_b{}s{}_{}.pdf".format(save_dir, b, s, save_note))

    @staticmethod
    def plot_pred_for_section(all_preds, all_labels, save_dir="./", section_num=100, save_note="example"):

        p = section_num
        sample_per_session = 20
        nrows, ncols = 4, 6
        total_num_session = int(all_preds['pred_speed'].shape[0] / sample_per_session)
        
        pred_by_session = np.split(all_preds['pred_speed'][:, -1, p], total_num_session)
        gt_by_session = np.split(all_labels['pred_speed'][:, -1, p], total_num_session)
        xx = np.arange(sample_per_session)
        
        sessions_to_include = list(range(int(total_num_session)))[-nrows * ncols:]

        fig, ax = plt.subplots(nrows, ncols, figsize=(13, 5))
        for i in range(nrows):
            for j in range(ncols):
                idx = i * ncols + j
                ax[i, j].plot(xx, pred_by_session[idx], label='30min_pred')
                ax[i, j].plot(xx, gt_by_session[idx], label='GT')
                ax[i, j].set_title("Sim. Session {}".format(idx), fontsize=0.8*plt.rcParams['font.size'])
        
        # add a common legend for all the subplots
        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncols=2)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the legend at the top
        fig.savefig("{}/30min_ahead_pred_{}_{}.pdf".format(save_dir, p, save_note))
        logger.info("Saved the plot to {}/30min_ahead_pred_{}_{}.pdf".format(save_dir, p, save_note))

    @staticmethod
    def plot_MAE_by_location(node_coordinates, all_preds, all_labels, save_dir=".F/", save_note="example"):
        MAE = torch.abs(all_preds['pred_speed'] - all_labels['pred_speed'])
        mae_by_section = torch.nanmean(MAE, dim=(0,1))
        fig, ax = plt.subplots(figsize=(6.5, 4))
        im = ax.scatter(node_coordinates[:, 0], node_coordinates[:, 1], c=mae_by_section.numpy(), s=2)
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("X Coordinates")
        ax.set_ylabel("Y Coordinates")
        fig.tight_layout()
        fig.savefig("{}/average_mae_{}.pdf".format(save_dir, save_note))
        logger.info("Saved the plot to {}/average_mae_{}.pdf".format(save_dir, save_note))
        
def delete_nodes(adj_mtx, delete_mask):
    if isinstance(adj_mtx, torch.Tensor):
        adj_mtx = adj_mtx.numpy()
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

    return torch.as_tensor(adj_mtx)

class SimBarcaRandomObservation(SimBarca):
    """ This class implements randomized drone observations and loop detector observations. The purpose is to make the dataset more realistic, because in reality, we don't have loop detectors for all road segments, and we can hardly fly enough drones to cover a whole city. Therefore, dealing with partial information is inevitable. 

    Concretely, 10% of the road segments will have loop detector observations, which will be initialized at the first time and later saved to file for reuse. In this way the loop detector positions are fixed across experiments, and the results can be fairly compared. 
    
    The drone observaions will be available for random 10% of the grid IDs (but not directly road segments), since we assume a drone to observe nearly all vehicles in its square-shaped FOV. To imitate a real-world scenario, each simulation session will be regarded as an individual day, when we fly the drones in different ways. We want the drone positions to mimic the flight plan for the simulation sessions, which will be consistent in all epochs in the same experiment and also across all experiments.
    
    While it's simple that the training and testing splits should share the same loop detector positions, the drone positions are more complicated. The training samples are generated in a sliding-window way based on the traffic statistics of the whole simulation. Therefore the drone positions should also be a "sliding-window", which means neighboring samples should have overlapped drone positions with only 1 time step difference. Check `load_or_init_drone_pos` for implementation. 
    
    Besides, for the training split, we exclude the data of the unmonitored road segments from both the input and output, which means partial input and partial label for the model to learn. The labels for regional speed predictions will also be created based on partial information (which makes it biased but that's the best we could do for now). However, for the test set, we still use the full information for evaluation (so don't train on the test set).
    """
    
    # we need them to recompute the regional speed values with the monitoring mask
    aux_seqs = ["pred_vdist", "pred_vtime"]
    
    def __init__(self, ld_cvg=0.1, drone_cvg=0.1, reinit_pos = False, random_state = 42, **kwargs):
        super().__init__(**kwargs)
        self.pred_vdist: torch.Tensor
        self.pred_vtime: torch.Tensor
        self.ld_cvg = ld_cvg
        self.drone_cvg = drone_cvg
        self.random_state = random_state
        self.reinit_pos = reinit_pos
        self.ld_mask: torch.Tensor
        self.drone_flight_mask: torch.Tensor
        self.load_or_init_ld_mask()
        self.load_or_init_drone_mask()
        if self.split == "train":
            self.add_seqs = ['ld_mask', 'drone_mask', 'pred_mask']
        elif self.split == "test":
            # output sequences are not masked in the test set, to test on the full information
            self.add_seqs = ['ld_mask', 'drone_mask']
    
    @property
    def ld_mask_file(self):
        return "{}/processed/ld_mask_{}.pkl".format(self.data_root, int(self.ld_cvg * 100))

    @property
    def drone_mask_file(self):
        return "{}/processed/drone_mask_{}_{}.pkl".format(self.data_root, int(self.drone_cvg*100), self.split)
    
    def load_or_init_ld_mask(self):
        if os.path.exists(self.ld_mask_file) and not self.reinit_pos:
            with open(self.ld_mask_file, "rb") as f:
                logger.info("Loading loop detector mask from file")
                ld_mask = pickle.load(f)
            if ld_mask.sum() != int(self.ld_cvg * self.adjacency.shape[0]):
                logger.warning("The loop detector coverage in the file are not consistent with the current settings, check `ld_cvg` argument or set `reinit_pos=True`")
        else:
            logger.info("Initializing loop detector mask using random seed {} and coverage {}".format(self.random_state, self.ld_cvg))
            rng = np.random.default_rng(self.random_state)
            # exclude the locations with too many nan, (which means these roads have insufficient vehicles)
            # and then randomly select a few indexes of the road segments to have loop detectors
            nan_by_location = np.mean(torch.isnan(self.ld_speed[..., 0]).numpy().astype(float), axis=(0, 1))
            # maybe it's OK to have 10% nan values 
            valid_pos = np.nonzero(nan_by_location < 0.1)[0] 
            if self.ld_cvg >= 1.0:
                ld_mask = np.ones(shape=self.adjacency.shape[0], dtype=bool)
            else:
                ld_pos = rng.choice(valid_pos, size=int(self.ld_cvg * self.adjacency.shape[0]), replace=False)
                ld_mask = np.zeros(shape=self.adjacency.shape[0], dtype=bool)
                ld_mask[ld_pos] = True
            
            with open(self.ld_mask_file, "wb") as f:
                pickle.dump(ld_mask, f)

        # the dtype must be bool, otherwise torch will regard it as indexes of the items to be selected
        self.ld_mask = torch.as_tensor(ld_mask, dtype=torch.bool)

    def load_or_init_drone_mask(self):
        
        if os.path.exists(self.drone_mask_file) and not self.reinit_pos:
            logger.info("Loading drone mask from file")
            all_drone_mask = pickle.load(open(self.drone_mask_file, "rb"))
            if np.abs(all_drone_mask.mean() - self.drone_cvg) > 0.05:
                logger.warning("The drone coverage in the file appears to be higher than config, check if `drone_cvg` argument has changed or set `reinit_pos=True`")
        else:
            logger.info("Initializing drone mask using random seed {} and coverage {}".format(self.random_state, self.drone_cvg))
            grid_cells = np.sort(np.unique(self.grid_id))
            rng = np.random.default_rng(self.random_state + 777) # avoid using the same random seed as ld_pos
            
            all_drone_mask = []
            for _ in range(len(self) // self.sample_per_session): # different simulation session
                # init drone positions for the first sample in each session
                # 30min input, 30min output, change every 3 mins, so that's 20 x num_grid 
                drone_mask = np.stack(
                    [np.isin(self.grid_id, 
                             rng.choice(grid_cells, size=int(self.drone_cvg * len(grid_cells)), replace=False)
                    ) for _ in range(20)]
                )
                all_drone_mask.append(drone_mask)
                # for every 3 min, sample a new set of drone positions and discard the earliest step
                for _ in range(1, self.sample_per_session):
                    next_step_drone_mask = np.isin(
                        self.grid_id, 
                        rng.choice(grid_cells, size=(int(self.drone_cvg * len(grid_cells))), replace=False)
                    )
                    drone_mask = np.concatenate((drone_mask[1:, :], next_step_drone_mask.reshape(1, -1)), axis=0)
                    all_drone_mask.append(drone_mask)
            all_drone_mask = np.stack(all_drone_mask, axis=0)
            with open(self.drone_mask_file, "wb") as f:
                pickle.dump(all_drone_mask, f)
        
        # the dtype must be bool, otherwise torch will regard it as indexes of the items to be selected
        self.drone_flight_mask = torch.as_tensor(all_drone_mask, dtype=torch.bool)

    def apply_masking(self, sample: Dict, ld_mask:torch.Tensor, drone_flight_mask: torch.Tensor) -> Dict:
        """ Apply the masking of loop detector and drone data to the input and label
        
            For both train and test sets:
                1. Set all INPUT modalities for unmonitored road segments to nan
            For train set only:
                1. Set all OUTPUT modalities for unmonitored road segments to nan
                2. Recalculate regional speed values based on the monitored road segments

            For the test set, we keep the output values as they are, because we want to evaluate the model's performance on the full information, even when the model is trained with partial data.
        """
        sample['ld_speed'][:, ld_mask == 0, 0] = torch.nan
        sample['ld_mask'] = ld_mask.bool()
        # the input and output sequences corresponds to a continuous 1 hour time window
        # and the drone_flight_mask is for this 1 hour. We take the first half hour for input 
        drone_mask = drone_flight_mask[:int(drone_flight_mask.shape[0]/2), :]
        # drone speeds are given every 5 seconds, but drones are assumed to change positions every 3 minutes
        # so we need to repeat the mask to match the time resolution
        drone_mask = torch.repeat_interleave(
                            drone_mask, 
                            int(sample['drone_speed'].shape[0]/drone_mask.shape[0]), 
                            dim=0)
        sample['drone_speed'][..., 0][drone_mask == 0] = torch.nan
        sample['drone_mask'] = drone_mask.bool()
        
        if self.split == "train":
            pred_mask = drone_flight_mask[int(drone_flight_mask.shape[0]/2):, :]
            sample['pred_speed'][..., 0][pred_mask == 0] = torch.nan
            sample['pred_mask'] = pred_mask
            
            # compute regional speed based on the monitored road segments
            regional_speed = []
            # aggregate link into regions
            for region_id in torch.unique(self.cluster_id):
                region_mask = torch.logical_and((self.cluster_id == region_id).reshape(1, -1), pred_mask)
                # sum the total distance but ignore NaN values, the sum will be NaN if one element is NaN
                region_vdist_values = torch.nansum(sample["pred_vdist"][..., 0]*region_mask.float(), dim=-1) 
                # add the time in day here, the first index
                # time in day was copied for all positions, so taking 1 is enough
                region_vdist_tind = sample["pred_vdist"][..., 0, 1]
                region_vtime_values = torch.nansum(sample["pred_vtime"][..., 0]*region_mask.float(), dim=-1)
                region_speed_values = region_vdist_values / region_vtime_values
                regional_speed.append(torch.stack([region_speed_values, region_vdist_tind], dim=-1))
            # the elements have shape (N, T, 2), where 2 corresponds to (time_in_day, value)
            # we stack them into shape (N, T, R, 2) where R is the number of regions
            sample["pred_speed_regional"] = torch.stack(regional_speed, dim=1)
            
        return sample
        
    def __getitem__(self, index):
        """ 
        In the files, we save the drone mask every 3 mins, since we assume the drones will move to monitor different locations every 3 minutes. However, the drone input is given every 5 seconds, so we need to extend the drone mask to the same time resolution as the drone input.
        
        The mask for loop detector is a 0-1 tensor of shape P, where P is the number of road segments. Since loop detectors are installed as fixed infrastructure, ld_mask remain unchanged over time.
        The mask for drone input of ONE sample is also a 0-1 tensor but it has shape (T, P), where P is the number of road segments, and T is the number of time steps. To avoid complex transition states, we assume drones to jump from one grid to another. 
        """
        data_dict = super().__getitem__(index)
        data_dict = self.apply_masking(data_dict, self.ld_mask, self.drone_flight_mask[index])
        
        # after applying the mask, we don't need the vehicle distance and time for training or testing
        del data_dict['pred_vdist']
        del data_dict['pred_vtime']
        
        return data_dict
    
        
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-scratch", action="store_true", help="Process everything from scratch")
    parser.add_argument("--filter-short", type=float, default=None, help="Filter out the short road segments")
    args = parser.parse_args()
    
    train_set = SimBarca(split="train", 
                         force_reload=args.from_scratch, 
                         filter_short=args.filter_short, 
                         use_clean_data=False)
    test_set = SimBarca(split="test", 
                        force_reload=args.from_scratch, 
                        filter_short=args.filter_short, 
                        use_clean_data=False)
    
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=train_set.collate_fn)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False, collate_fn=test_set.collate_fn)

    for data_dict in train_loader:
        SimBarca.visualize_batch(data_dict, save_note="train")
        break

    for data_dict in test_loader:
        SimBarca.visualize_batch(data_dict, save_note="test")
        break

    debug_set = SimBarcaRandomObservation(split='train', reinit_pos=args.from_scratch, random_state=42)
    sample = debug_set[0]
    batch = debug_set.collate_fn([debug_set[0], debug_set[100]])
    
    debug_set = SimBarcaRandomObservation(split='test', reinit_pos=args.from_scratch, random_state=42)
    sample = debug_set[0]
    batch = debug_set.collate_fn([debug_set[0], debug_set[100]])
