import torch
import numpy as np
from typing import Dict, List

from .simbarca_base import SimBarcaForecast

class SimBarcaSpeed(SimBarcaForecast):
    """ Simulated speed data for Barcelona. Input window 30 minutes, output window 30 minutes, with 3 minutes step size.
        The invalid values are marked as nan, which is usually not a problem for the label, as most loss functions
        are able to handle nan values. However, for the input, many models are not built with a mechanism to handle nan values,
        because in previous literature, they are replaced with zeros. 
        We avoid using zeros for a missing value because in urban traffic, a zero speed is valid when all vehicles are stopped.

    Args:
        split (str, optional): train or test. Defaults to "train".
        input_nan_to_global_avg (bool, optional): if True, replace the nan values with the global average. Defaults to True.
    """
    input_steps: int = 10
    pred_steps: int = 10
    num_nodes: int = 1570
    data_null_value: float = float("nan")

    def __init__(self, split="train", input_nan_to_global_avg:bool=True):
        super().__init__(split, input_window=30, pred_window=30, step_size=3, sample_per_session=20)
        self.input_window: int
        self.pred_window: int
        self.step_size: int
        self.sample_per_session: int
        self.input_nan_to_global_avg: bool = input_nan_to_global_avg
        self.global_avg: float = 0.0

        self.time_in_day: torch.Tensor # shape (T)
        self.speed_data: torch.Tensor # shape (n_sessions, T, P)
        self.in_indexes: torch.Tensor # boolean tensor of shape (sample_per_session, T)
        self.out_indexes: torch.Tensor # boolean tensor of shape (sample_per_session, T)

        self.prepare_data_for_prediction()
        self.load_or_compute_metadata()
        self.clean_up_raw_sequences()

    def __repr__(self):
        return f"""SimBarcaSpeed(
            split={self.split},
            input_window={self.input_window},
            pred_window={self.pred_window},
            step_size={self.step_size},
            sample_per_session={self.sample_per_session}
        )"""
    
    def __len__(self):
        return self.sample_per_session * self.num_sessions
    
    def __getitem__(self, index):
        """ get the time slices from the input and output, for both 5s and 3min data
            and then get time series data from the data sequences using the time slices
        """
        session_id = index // self.sample_per_session
        offset = index % self.sample_per_session

        # for input, we need to encode the time of the day for more information, no need for output 
        input_time_in_day = self.time_in_day[self.in_indexes[offset]]
        input_values = self.speed_data[session_id, self.in_indexes[offset], :]
        output_values = self.speed_data[session_id, self.out_indexes[offset], :]
        if self.input_nan_to_global_avg:
            input_values = torch.nan_to_num(input_values, nan=self.global_avg)

        # repeat the time in day over the spatial locations and stack with the speed data 
        source = torch.stack([input_values, input_time_in_day.unsqueeze(1).repeat(1, input_values.shape[1])], dim=-1)

        return {
            "source": source,
            "target": output_values,
        }

    def prepare_data_for_prediction(self):
        # convert the time stamp to time in day, e.g., 9am will be 9 hour/24 hour = 0.375
        self.time_in_day = (self._timestamp_3min - self._timestamp_3min[0].astype('datetime64[D]')) / np.timedelta64(24, "h")
        # derive speed data from distance and time
        self.speed_data = self._vdist_3min / self._vtime_3min
        self.in_indexes, self.out_indexes = self.get_sample_in_out_index(self._timestamp_3min)
        if self.input_nan_to_global_avg:
            self.global_avg = np.nanmean(self.speed_data)

        # convert to torch tensors when everything is ready
        self.time_in_day = torch.from_numpy(self.time_in_day).to(torch.float32)
        self.speed_data = torch.from_numpy(self.speed_data).to(torch.float32)
        self.in_indexes = torch.from_numpy(self.in_indexes).to(torch.bool)
        self.out_indexes = torch.from_numpy(self.out_indexes).to(torch.bool)

    def load_or_compute_metadata(self):
        metadata = {
            "input_steps": self.input_steps,
            "pred_steps": self.pred_steps,
            "num_nodes": self.num_nodes,
            "adjacency": torch.as_tensor(self.adjacency, dtype=torch.long),
            "edge_index": torch.as_tensor(self.edge_index, dtype=torch.long),
            "node_coordinates": torch.as_tensor(self.node_coordinates, dtype=torch.float32)
        }

        # calculate the mean and std of the speed data, excluding the nan values
        metadata['mean'] = torch.mean(self.speed_data[~torch.isnan(self.speed_data)]).item()
        metadata['std'] = torch.std(self.speed_data[~torch.isnan(self.speed_data)]).item()
        metadata['data_dim'] = 0
        
        self.metadata = metadata

    def collate_fn(self, list_of_data_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            "source": torch.stack([data_dict["source"] for data_dict in list_of_data_dicts], dim=0),
            "target": torch.stack([data_dict["target"] for data_dict in list_of_data_dicts], dim=0),
        }
    
