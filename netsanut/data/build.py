from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from typing import List, Tuple, Dict
from .catalog import DATASET_CATALOG

def build_dataset(cfg: Dict):
    """ instantiate a dataset object, and configurations are passed as kwargs
    """
    dataset_cfg = deepcopy(cfg)
    dataset_name = dataset_cfg.pop('name')
    # support for multiple datasets can be added here if necessary
    dataset = DATASET_CATALOG[dataset_name](**dataset_cfg)
    return dataset 

def build_train_loader(dataset: Dataset, cfg: Dict):
    loader_cfg = deepcopy(cfg)
    shuffle = loader_cfg.pop('shuffle', True) # default to True
    dataloader = DataLoader(dataset, shuffle=shuffle, collate_fn=dataset.collate_fn, **loader_cfg)
    return dataloader

def build_test_loader(dataset, cfg: Dict):
    """ create a test loader, the main differences from the train loader are:
            1. no data shuffling by default
            2. no sampler will be used by default
    """
    loader_cfg = deepcopy(cfg)
    shuffle = loader_cfg.pop('shuffle', False) # default to False
    dataloader = DataLoader(dataset, shuffle=shuffle, collate_fn=dataset.collate_fn, **loader_cfg)
    return dataloader