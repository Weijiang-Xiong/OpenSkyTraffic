import os
import json
import tqdm
import logging
import pickle

from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns 

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import skytraffic

from .simbarca_base import SimBarcaForecast
 
sns.set_style("darkgrid")

_package_init_file = skytraffic.__file__
_root: Path = (Path(_package_init_file).parent.parent).resolve()
assert _root.exists(), "please check package installation"

logger = logging.getLogger("default")

class SimBarcaMSMT(SimBarcaForecast):
    
    """ Invalid values in the dataset are represented as NaN, clear error occurs if NaNs are not properly handled, e.g., one will see NaN in the loss and backpropagation will fail.
    Instead, if invalid values are replaced by indicators like -1, the computation can still go on without problem and the results will contain a silent error, which is more difficult to discover. 
    """

    input_seqs = ["drone_speed", "ld_speed"] # input sequences to feed to the model, will be normalized 
    output_seqs = ["pred_speed", "pred_speed_regional"] # output sequences to predict, will be normalized
    
    def __init__(self, split="train", input_window=30, pred_window=30, step_size=3, sample_per_session=20):
        super().__init__(split, input_window, pred_window, step_size, sample_per_session)

        # data sequences in the dataset 
        self.timestamp_5s: torch.Tensor # shape (T_high, ) where T_high is a big value for high temporal resolution
        self.drone_speed: torch.Tensor # shape (num_sessions, T_high, P) where P is the number of spatial locations
        self.timestamp_3min: torch.Tensor # shape (T_low, ) where T_low is a small value for low temporal resolution
        self.ld_speed: torch.Tensor # shape (num_sessions, T_low, P)
        self.pred_speed: torch.Tensor # shape (num_sessions, T_low, P)
        self.pred_speed_regional: torch.Tensor # shape (num_sessions, T_low, R) where R is the number of regions

        self.prepare_data_for_prediction()
        self.load_or_compute_metadata()
        self.clean_up_raw_sequences()

    @property
    def io_seqs(self):
        """ all the traffic variable series that are either input to model or required to predict
        """
        return self.input_seqs + self.output_seqs
    
    @property
    def metadata_file(self):
        return "{}/{}.pkl".format(self.metadata_folder, self.__class__.__name__.lower())

    def __len__(self):
        return self.num_sessions * self.sample_per_session

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        """ get the time slices from the input and output, for both 5s and 3min data
            and then get time series data from the data sequences using the time slices
        """
        session_id = index // self.sample_per_session
        sample_id = index % self.sample_per_session
        
        # input drone speed and time in day encoding, every 5s 
        in_tid_5s = self.time_in_day_5s[self.in_indexes_5s[sample_id]]
        in_drone = self.drone_speed[session_id, self.in_indexes_5s[sample_id], :]
        drone_speed = torch.stack([in_drone, in_tid_5s.unsqueeze(1).repeat(1, in_drone.shape[1])], dim=-1)

        # input ld speed and time in day encoding, every 3min 
        in_tid_3min = self.time_in_day_3min[self.in_indexes_3min[sample_id]]
        in_ld = self.ld_speed[session_id, self.in_indexes_3min[sample_id], :]
        ld_speed = torch.stack([in_ld, in_tid_3min.unsqueeze(1).repeat(1, in_ld.shape[1])], dim=-1)

        # prediction targets, segment-level speed prediction and regional speed prediction, every 3min 
        pred_speed = self.pred_speed[session_id, self.out_indexes_3min[sample_id], :]
        pred_speed_regional = self.pred_speed_regional[session_id, self.out_indexes_3min[sample_id], :]

        return {
            "drone_speed": drone_speed,
            "ld_speed": ld_speed,
            "pred_speed": pred_speed,
            "pred_speed_regional": pred_speed_regional,
        }


    def prepare_time_steps(self):
        # Convert timestamps to time_in_day (normalized to 0-1 range) for 5s and 3min
        self.time_in_day_5s = (self._timestamp_5s - self._timestamp_5s[0].astype('datetime64[D]')) / np.timedelta64(24, "h")
        self.in_indexes_5s, self.out_indexes_5s = self.get_sample_in_out_index(self._timestamp_5s)
        self.time_in_day_3min = (self._timestamp_3min - self._timestamp_3min[0].astype('datetime64[D]')) / np.timedelta64(24, "h")
        self.in_indexes_3min, self.out_indexes_3min = self.get_sample_in_out_index(self._timestamp_3min)
        self.time_in_day_5s = torch.from_numpy(self.time_in_day_5s).to(torch.float32)
        self.in_indexes_5s = torch.from_numpy(self.in_indexes_5s).to(torch.bool)
        self.out_indexes_5s = torch.from_numpy(self.out_indexes_5s).to(torch.bool)
        self.time_in_day_3min = torch.from_numpy(self.time_in_day_3min).to(torch.float32)
        self.in_indexes_3min = torch.from_numpy(self.in_indexes_3min).to(torch.bool)
        self.out_indexes_3min = torch.from_numpy(self.out_indexes_3min).to(torch.bool)

    def prepare_data_for_prediction(self):
        self.prepare_time_steps()

        # Derive drone speed from distance and time
        self.drone_speed = self._vdist_5s / self._vtime_5s
        # ld_speed is already speed data
        self.ld_speed = self._ld_speed_3min
        # segment speed derived from distance and time
        self.pred_speed = self._vdist_3min / self._vtime_3min
        # regional speed computed from segment level data
        self.pred_speed_regional = self.regional_speed_from_segment_numpy(self._vdist_3min, self._vtime_3min)
        
        # Convert to torch tensors when data sequences are ready
        self.drone_speed = torch.from_numpy(self.drone_speed).to(torch.float32)
        self.ld_speed = torch.from_numpy(self.ld_speed).to(torch.float32)
        self.pred_speed = torch.from_numpy(self.pred_speed).to(torch.float32)
        self.pred_speed_regional = torch.from_numpy(self.pred_speed_regional).to(torch.float32)


    def load_or_compute_metadata(self) -> Dict[str, Any]:
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "rb") as f:
                metadata = pickle.load(f)
                logger.info("Loaded metadata from {}, containing {}".format(self.metadata_file, metadata.keys()))
        else:
            if self.split != "train":
                logger.warning("Metadata should be computed using the train set, please instantiate the train set to compute and save the metadata ...")

            metadata = {
                "adjacency": torch.as_tensor(self.adjacency, dtype=torch.long),
                "edge_index": torch.as_tensor(self.edge_index, dtype=torch.long),
                "cluster_id": torch.as_tensor(self.cluster_id, dtype=torch.long),
                "grid_id": torch.as_tensor(self.grid_id, dtype=torch.long),
            }
            
            mean_and_std = dict()
            for att in self.io_seqs:
                seq_data = getattr(self, att)
                seq_data = seq_data[~torch.isnan(seq_data)]
                mean_and_std[att] = (torch.mean(seq_data), torch.std(seq_data))
            metadata["mean_and_std"] = mean_and_std

            metadata["input_seqs"] = self.input_seqs
            metadata["output_seqs"] = self.output_seqs

            # save the metadata to a file
            with open(self.metadata_file, "wb") as f:
                pickle.dump(metadata, f)

        self.metadata = metadata
    
    def regional_speed_from_segment_numpy(self, seg_vdist, seg_vtime) -> np.ndarray:
        """ compute regional speed from segment level vehicle travel distance and travel time
        """
        regional_speed_list = []
        # aggregate link into regions
        for region_id in np.unique(self.cluster_id):
            region_mask = self.cluster_id == region_id
            # sum the total distance but ignore NaN values, `np.sum` will be NaN if one element is NaN
            region_vdist_values = np.nansum(seg_vdist[..., region_mask], axis=-1)
            region_vtime_values = np.nansum(seg_vtime[..., region_mask], axis=-1)
            region_speed_values = region_vdist_values / region_vtime_values
            # the regional speed is very unlikely to be nan, but we still don't exclude this possibility
            # region_speed_values = np.nan_to_num(region_speed_values, nan=-1)
            regional_speed_list.append(region_speed_values)
        # the elements have shape (N, T) where N is the number of sessions and T is the number of time steps
        # we stack them into shape (N, T, R) where R is the number of regions
        regional_speed_array = np.stack(regional_speed_list, axis=2)

        return regional_speed_array

    def collate_fn(self, list_of_seq: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_data = dict()
        for attr in self.io_seqs:
            batch_data[attr] = torch.cat(
                [seq[attr].unsqueeze(0) for seq in list_of_seq], dim=0
            ).contiguous()

        batch_data["metadata"] = self.metadata

        return batch_data


class SimBarca(Dataset):
    
    """ Invalid values in the dataset are represented as NaN, clear error occurs if NaNs are not properly handled, e.g., one will see NaN in the loss and backpropagation will fail.
    Instead, if invalid values are replaced by indicators like -1, the computation can still go on without problem and the results will contain a silent error, which is more difficult to discover. 
    """

    data_root = "datasets/simbarca"
    metadata_folder = "{}/metadata".format(data_root)
    eval_metrics_folder = "{}/eval_metrics".format(data_root)
    session_splits = "{}/train_test_split.json".format(metadata_folder)
    session_folder_pattern = "simulation_sessions/session_*"
    soi_file = "{}/sections_of_interest.txt".format(metadata_folder)
    sample_per_session = 20 # the number of samples in each simulation session
    
    input_seqs = ["drone_speed", "ld_speed"] # input sequences to feed to the model
    output_seqs = ["pred_speed", "pred_speed_regional"] # output sequences to predict
    add_seqs = [] # additional sequences that are required in the input batch, not time series of traffic variables (like the monitoring mask in SimbarcaRandomObservation)
    aux_seqs = [] # sequences that are required during data processing, they are traffic variables but are not input or output, needed in collate but deleted afterwards (vkt and vht for recomputing regional label)
    
    def __init__(self, split="train", force_reload=False):
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
        self.section_ids_sorted: torch.Tensor
        self.section_id_to_index: Dict[int, int]
        self.index_to_section_id: Dict[int, int]
        self.session_ids: torch.Tensor # shape (K, ) where K is the number of simulation sessions
        self.demand_scales: torch.Tensor # shape (K, ) where K is the number of simulation sessions
        
        samples = self.load_or_process_samples()
        for attribute in self.io_seqs + self.aux_seqs:
            attr_data = torch.as_tensor(samples[attribute], dtype=torch.float32)
            setattr(self, attribute, attr_data)
            del samples[attribute] # free memory otherwise 64 GB won't be enough ... 
        self.session_ids, self.demand_scales = self.get_session_properties()
        
        self.data_augmentations = []
        
        # load the sections of interest to visualize segment-level predictions
        with open(self.soi_file, "r") as f:
            self.sections_of_interest = [int(x) for x in f.read().split(",")]
        
        self.metadata = self.load_or_compute_metadata()

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        # don't use tuple comprehension here, otherwise it will return a generator instead of actual data
        data_seqs = {attr: getattr(self, attr)[index] for attr in self.io_seqs + self.aux_seqs}

        return data_seqs

    def __len__(self):
        return self.drone_speed.shape[0]

    @property
    def io_seqs(self):
        """ all the traffic variable series that are either input to model or required to predict
        """
        return self.input_seqs + self.output_seqs
    
    @property
    def processed_file(self):
        return "{}/processed/{}.npz".format(self.data_root, self.split)

    def get_sessions_in_split(self) -> List[Path]:
        """ return a list of paths that contains the simulation sessions in the split
        """
        
        if not os.path.exists(self.session_splits):
            print("No train_test_split.json file found, please use `preprocess/simbarca/choose_train_test.py`")
        with open(self.session_splits, "r") as f:
            session_ids = json.load(f)[self.split]

        sessions_in_split = [Path(
            "{}/{}".format(self.data_root, self.session_folder_pattern).replace("*", "{:03d}".format(x))
            ).absolute() for x in sorted(session_ids)]
        
        return sessions_in_split

    def load_or_compute_metadata(self):
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

    def get_session_properties(self):
        
        def session_number_from_path(path):
            import re
            return int(re.search(r"session_(\d+)", str(path)).group(1))
        
        sessions_in_split = self.get_sessions_in_split()
            
        session_ids, demand_scales = [], []
        for f in sessions_in_split:
            scale = json.load(open("{}/settings.json".format(f), 'r'))["global_scale"]
            session_id = session_number_from_path(f)
            session_ids.append(session_id)
            demand_scales.append(scale)
        
        return torch.as_tensor(session_ids), torch.as_tensor(demand_scales)
        
    def read_graph_structure(self):
        """ read the graph structure for the road network from Aimsun-exported metadata.
        """
        folder = "{}/{}".format(_root, self.metadata_folder)
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
        self.section_ids_sorted: torch.Tensor = torch.as_tensor(section_ids_sorted)
        self.index_to_section_id: Dict = index_to_section_id
        self.section_id_to_index: Dict = section_id_to_index

    
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
            logger.info("Trying to load existing processed samples for {} split".format(self.split))
            with open(self.processed_file, "rb") as f:
                loaded_data = np.load(f)
                return {key: value for key, value in loaded_data.items() if key in self.io_seqs + self.aux_seqs}
        else:
            logger.info("No processed samples found or forced to reload, processing samples from scratch")
            return self.process_samples()
        
    def process_samples(self) -> List[Dict[str, pd.DataFrame | np.ndarray]]:
        
        all_sample_data = defaultdict(list)
        split_sample_files =["{}/timeseries/samples.npz".format(f) for f in self.get_sessions_in_split()]

        with open(self.session_splits, "r") as f:
            session_ids = json.load(f)[self.split]
        logger.info("The simulation sessions in the {} split are {}".format(self.split, session_ids))
        
        logger.info("Found {} samples for {} split, reading them one by one".format(len(split_sample_files), self.split))
        for sample_file in tqdm.tqdm(split_sample_files):
            with open(sample_file, "rb") as f:
                sample_data: np.lib.npyio.NpzFile = np.load(f)
                for key in sample_data:
                    all_sample_data[key].append(sample_data[key])

        # concatenate the samples along the batch dimension
        for key, value in all_sample_data.items():
            all_sample_data[key] = np.concatenate(value, axis=0)

        logger.info("Processing per section data")
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
        
        logger.info("Processing regional data")
        processed_samples["pred_speed_regional"] = self.regional_speed_from_segment(
            all_sample_data["pred_vdist"], all_sample_data["pred_vtime"]
        )

        # we also keep the vehicle distances and time for the drone
        processed_samples["drone_vdist"] = all_sample_data["drone_vdist"]
        processed_samples["pred_vdist"] = all_sample_data["pred_vdist"]
        processed_samples["drone_vtime"] = all_sample_data["drone_vtime"]
        processed_samples["pred_vtime"] = all_sample_data["pred_vtime"]

        # save all samples as a compressed npz file
        logger.info("Saving processed samples to {}".format(self.processed_file))
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
        


