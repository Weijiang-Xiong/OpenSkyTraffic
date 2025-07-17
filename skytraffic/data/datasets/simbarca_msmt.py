import os
import logging
import pickle
from typing import List, Dict, Any

import numpy as np

import torch
from .simbarca_base import SimBarcaForecast
 
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