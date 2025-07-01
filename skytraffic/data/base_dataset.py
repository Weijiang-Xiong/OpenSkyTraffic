from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Dict, Any, List

class BaseDataset(Dataset, ABC):
    """
    This is a base class for all datasets, it does nothing but defining the interface.
    """
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