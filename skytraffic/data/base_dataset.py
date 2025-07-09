from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Dict, Any, List

class BaseDataset(Dataset, ABC):
    """
    This is a base class for all datasets, it does nothing but defining the interface.
    We assume a dataset should have the following attributes for creating a model:
        - input_steps: the number of timesteps to use as input
        - pred_steps: the number of timesteps to predict
        - num_nodes: the number of nodes in the graph
        - data_null_value: the value corresponding to missing data
        - metadata: a dictionary of metadata for the dataset

    In case of multiple input or output sources or a compound dataset, the attributes should be a list of values.
    """

    input_steps: int
    pred_steps: int
    num_nodes: int
    data_null_value: float
    metadata: Dict[str, Any]
    
    @abstractmethod
    def __len__(self) -> int:
        pass


    @abstractmethod
    def __getitem__(self, idx: int):
        pass


    @abstractmethod
    def load_or_compute_metadata(self) -> Dict[str, Any]:
        pass


    def collate_fn(self, list_of_data_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        pass


    def summarize(self) -> Dict[str, Any]:
        pass