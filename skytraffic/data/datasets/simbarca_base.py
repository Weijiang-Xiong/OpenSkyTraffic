import os
import json
import logging
import datetime
import pickle
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from ..base_dataset import BaseDataset

logger = logging.getLogger("default")

class SimbarcaBase(BaseDataset):
    """
    This dataset implements the functions that are useful at the level of simulation sessions.
    Data are stored as numpy arrays (not pandas dataframes or torch tensors).

        1. reading the aggregated timeseries, and reading the simulation settings by sessions
        2. reading the road network structure and setting the metadata (these are provided by simulator)
        3. assigning the aggregated timeseries to the attributes of the dataset

    This class is not meant to be used directly, so the required methods are still abstractmethods.
    Trying to initialize this class will raise an error.

    ==== Below are implementation details ====

    In step 3, the aggregated timeseries are filtered to only include the time range specified by `_sample_start_time` and `_sample_end_time`.
    This is because in the simulations, the travel demand has a warm-up from 7:45 to 8:00, and then the main demand ends at 9:45.
    Afterwards, the simulation has a clearing-out period until 11:45 (4 hours in total), this is to make sure that by the end of the simulation, 
    the vehicles have enough time to leave the road network (provided that there is no gridlock).
    However, when the road network does not have enough vehicles, the simulated traffic will be very unrealistic, and therefore we need to avoid
    such time periods, and only use the time range from 8:00 to 10:00 (this can be adjusted by changing the start and end time in this class)
    
    When calculating these timeseries, the time step is the end time of the aggregation interval, e.g., the statistics from 7:57:00 to 8:00:00 will be assigned to 8:00.
    Therefore, the timestamp of the first sample is 8:03 up to 9:00, and we have 20 samples per session, so the last one has timestamp from 8:57 to 9:57.
    One can easily reduce the number of samples by setting an later _sample_start_time or an earlier _sample_end_time, but remember to keep aligned with 3-min step size.
    E.g., you can start at 8:06 but not 8:07, because the step size is 3 minutes, and it starts from 7:45.
    """

    data_root = "datasets/simbarca"
    metadata_folder = "{}/metadata".format(data_root)
    session_splits = "{}/train_test_split.json".format(metadata_folder)
    session_folder_pattern = "simulation_sessions/session_*"
    soi_file = "{}/sections_of_interest.txt".format(metadata_folder)
    _sample_start_time = np.datetime64("2005-05-10T08:00:00") # so this is the day set in the simulator
    _sample_end_time = np.datetime64("2005-05-10T10:00:00")
    data_null_value = float("nan")

    def __init__(self, split="train"):
        self.split = split

        # data sequences along with timestamps 
        # the provided raw sequences are vehicle travel distance (vdist) and travel time (vtime) for all locations (road segments)
        # taggregated every 5 seconds and every 3 minutes, corresponding to high-frequency drones and low-frequency loop detectors
        self._timestamp_5s: np.ndarray # shape (T_high,)
        self._vdist_5s: np.ndarray # shape (num_sessions, T_high, num_locations)
        self._vtime_5s: np.ndarray # shape (num_sessions, T_high, num_locations)
        self._timestamp_3min: np.ndarray # shape (T_low,)
        self._ld_speed_3min: np.ndarray # shape (num_sessions, T_low, num_locations)
        self._vdist_3min: np.ndarray # shape (num_sessions, T_low, num_locations)
        self._vtime_3min: np.ndarray # shape (num_sessions, T_low, num_locations)
        
        # simulation session info, for grouped evaluation 
        self.session_ids: List[int] 
        self.demand_scales: List[float]

        # graph structure 
        self.adjacency: np.ndarray
        self.segment_lengths: np.ndarray
        self.edge_index: np.ndarray
        self.node_coordinates: np.ndarray
        self.cluster_id: np.ndarray
        self.grid_id: np.ndarray
        self.section_ids_sorted: np.ndarray
        self.section_id_to_index: Dict[int, int]
        self.index_to_section_id: Dict[int, int]
        self.intersection_polygon: Dict[str, Any]

        # initialize data sequences and metadata 
        self.init_raw_sequences(self.load_seq_files())
        self.init_graph_structure()
        session_info = self.get_session_properties()
        self.session_ids = session_info["session_ids"]
        self.demand_scales = session_info["demand_scales"]

        # a few representative sections, for visualization examples 
        with open(self.soi_file, "r") as f:
            self.sections_of_interest = [int(x) for x in f.read().split(",")]

    @property
    def num_sessions(self) -> int:
        return len(self.session_ids)

    def load_seq_files(self) -> List[Dict[str, Any]]:
        """Load agg_timeseries.pkl files from various session folders in parallel"""
        
        sessions_in_split = self.get_sessions_in_split()
        if not sessions_in_split:
            logger.warning("No sessions found for split '{}'".format(self.split))
            return []
            
        sample_files = ["{}/timeseries/agg_timeseries.pkl".format(f) for f in sessions_in_split]
        logger.info("Loading {} sample files for {} split in parallel".format(len(sample_files), self.split))
        
        def load_file(file_path):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        
        with ThreadPoolExecutor() as executor:
            loaded_seqs: List[pd.DataFrame] = list(executor.map(load_file, sample_files))

        return loaded_seqs

    def init_raw_sequences(self, seqs: List[pd.DataFrame]):

        # drone measurements, every 5s
        timestamp_5s: pd.DatetimeIndex = seqs[0]['drone_vdist'].index.to_numpy()
        start_index_5s = np.where(timestamp_5s == self._sample_start_time)[0][0] + 1 # do not include 8:00 
        end_index_5s = np.where(timestamp_5s == self._sample_end_time)[0][0] # do not include 10:00 
        self._timestamp_5s = timestamp_5s[start_index_5s:end_index_5s]
        self._vdist_5s = np.stack([seq['drone_vdist'].iloc[start_index_5s:end_index_5s].to_numpy() for seq in seqs], axis=0)
        self._vtime_5s = np.stack([seq['drone_vtime'].iloc[start_index_5s:end_index_5s].to_numpy() for seq in seqs], axis=0)

        # loop detector measurements, every 3 minutes 
        timestamp_3min: pd.DatetimeIndex = seqs[0]['pred_vtime'].index.to_numpy()
        start_index_3min = np.where(timestamp_3min == self._sample_start_time)[0][0] + 1 # do not include 8:00 
        end_index_3min = np.where(timestamp_3min == self._sample_end_time)[0][0] # do not include 10:00 
        self._timestamp_3min = timestamp_3min[start_index_3min:end_index_3min]
        self._ld_speed_3min = np.stack([seq['ld_speed'].iloc[start_index_3min:end_index_3min].to_numpy() for seq in seqs], axis=0)

        # sequences for constructing prediction targets, every 3 minutes 
        self._vdist_3min = np.stack([seq['pred_vdist'].iloc[start_index_3min:end_index_3min].to_numpy() for seq in seqs], axis=0)
        self._vtime_3min = np.stack([seq['pred_vtime'].iloc[start_index_3min:end_index_3min].to_numpy() for seq in seqs], axis=0)
        

    def clean_up_raw_sequences(self):
        # this is for cleaning up the raw sequences after obtaining required ones in a subclass
        del self._vdist_5s
        del self._vtime_5s
        del self._ld_speed_3min
        del self._vdist_3min
        del self._vtime_3min
        del self._timestamp_5s
        del self._timestamp_3min

    def get_sessions_in_split(self) -> List[Path]:
        """Return a list of paths that contains the simulation sessions in the split"""
        
        if not os.path.exists(self.session_splits):
            print("No train_test_split.json file found, please use `preprocess/simbarca/choose_train_test.py`")
            return []
            
        with open(self.session_splits, "r") as f:
            session_ids = json.load(f)[self.split]

        sessions_in_split = [Path(
            "{}/{}".format(self.data_root, self.session_folder_pattern).replace("*", "{:03d}".format(x))
            ).absolute() for x in sorted(session_ids)]
        
        return sessions_in_split


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
        
        return {
            "session_ids": session_ids,
            "demand_scales": demand_scales
        }


    def init_graph_structure(self):
        """ read the graph structure for the road network from Aimsun-exported metadata.
        """
        
        connections = pd.read_csv(
            "{}/connections.csv".format(self.metadata_folder),
            dtype={
                "turn": int,
                "org": int,
                "dst": int,
                "intersection": int,
                "length": float,
            },
        )
        link_bboxes = pd.read_csv(
            "{}/link_bboxes_clustered.csv".format(self.metadata_folder),
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
        with open("{}/intersec_polygon.json".format(self.metadata_folder), "r") as f:
            intersection_polygon = json.load(f)

        link_bboxes = link_bboxes.sort_values(by=["id"])
        section_ids_sorted = link_bboxes["id"].to_numpy()
        section_id_to_index = {link_id.item(): index for index, link_id in enumerate(section_ids_sorted)}
        index_to_section_id = {index: section_id.item() for index, section_id in enumerate(section_ids_sorted)}
        cluster_id = link_bboxes["cluster"].to_numpy()  # check with the csv file
        section_grid_id = link_bboxes["grid_nb"].to_numpy()  # check with the csv file
        node_coordinates = link_bboxes[["c_x", "c_y"]].to_numpy()
        segment_lengths = link_bboxes["length"].to_numpy()
        num_lanes = link_bboxes["num_lanes"].to_numpy()
        
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
        self.cluster_id = cluster_id
        self.grid_id = section_grid_id
        self.section_ids_sorted = section_ids_sorted
        self.index_to_section_id = index_to_section_id
        self.section_id_to_index = section_id_to_index
        self.intersection_polygon = intersection_polygon
        self.num_lanes = num_lanes


class SimBarcaForecast(SimbarcaBase):

    """ This class implements the essential functions for forecasting tasks.
            1. determine how many samples are available, based on the valid simulation duration and the input/output window size
            2. compute the time slices for the input and outputs, which are used to index the timestamps and data sequences
    """

    def __init__(self, split="train", input_window=30, pred_window=30, step_size=3, sample_per_session=20):
        super().__init__(split)
        self.input_window = input_window
        self.pred_window = pred_window
        self.step_size = step_size

        # only part of the simulation is used, which puts a limit to the number of samples
        valid_sim_duration = int((self._sample_end_time - self._sample_start_time) / np.timedelta64(1, 'm'))
        max_num_samples = 1 + (valid_sim_duration - (self.input_window + self.pred_window)) // self.step_size
        self.sample_per_session = min(max_num_samples, sample_per_session)

    def get_sample_in_out_index(self, timestamp: np.ndarray):
        """ This function pre-computes the time indexes for the input and output of each training sample.
            We generate `sample_per_session` samples from each simulation session, and each sample has its own input time window and output time window.
            The aim of this function is to return a True/False indicator for the inputs and outputs of each sample, so that one can use it to index the data sequences.
                ```
                self.data_sequence[sim_session_id][in_indexes[offset]]
                ```
        """
        in_indexes, out_indexes = [], []

        # pre_compute_time_slices
        for offset in range(self.sample_per_session):
            # the starting, middle and ending time of the sliding window 
            t_s = self._sample_start_time + np.timedelta64(self.step_size * offset, "m")
            t_m = t_s + np.timedelta64(self.input_window, "m")
            t_e = t_m + np.timedelta64(self.pred_window, "m")
            # the time steps for the data in the input time window (t_s, t_m)
            in_index = (timestamp > t_s) & (timestamp <= t_m)

            # the time steps for the data in the output time window (t_m, t_e)
            out_index = (timestamp > t_m) & (timestamp <= t_e)

            in_indexes.append(in_index)
            out_indexes.append(out_index)

        # convert to numpy arrays and return
        return np.array(in_indexes), np.array(out_indexes)

