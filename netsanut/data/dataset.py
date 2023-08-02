import os
import logging
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np 

import torch 
from torch.utils.data import Dataset, DataLoader

from .catalog import DATASET_CATALOG
from .adjacency import load_adjacency

logger = logging.getLogger('default')

class NetworkedTimeSeriesDataset(Dataset):
    """ a dataset class for time series on a bunch of nodes (e.g., traffic 
    flow in a city)
    """
    
    def __init__(self, name:str='metr-la', split='train', compute_metadata=False, adj_type='doubletransition') -> None:
        super().__init__()
        self.name, self.split = name, split
        data = np.load(DATASET_CATALOG[self.name][self.split])
        # data assumed to have (N, T, M, C) shape
        self.data_x = torch.as_tensor(data['x'], dtype=torch.float32)
        self.data_y = torch.as_tensor(data['y'], dtype=torch.float32)
        
        if self.split == 'train' or compute_metadata:
            self.metadata = self.compute_metadata(adj_type)

    def __len__(self) -> int:
        return int(self.data_x.size(0))
    
    def __getitem__(self, index: int):
        return self.data_x[index], self.data_y[index]
    
    def get_geo_locations(self):
        match self.name:
            case 'metr-la':
                return pd.read_csv(DATASET_CATALOG[self.name]['geo_locations'])[['longitude', "latitude"]].to_numpy()
            case 'pems-bay':
                return pd.read_csv(DATASET_CATALOG[self.name]['geo_locations'], header=None)[[2,1]].to_numpy()

    def compute_metadata(self, adj_type):
        sensor_ids, sensor_id_to_ind, adj = load_adjacency(DATASET_CATALOG[self.name]['adjacency'], adj_type)
        adjacency_matrix: List[torch.Tensor] = [torch.as_tensor(A) for A in adj]
        # separately calculate the mean and std for each channel
        # although it doesn't make sense to do this for time, we can handle it later
        data_mean = torch.mean(self.data_x, dim=list(range(self.data_x.dim()))[:-1])
        data_std  = torch.std(self.data_x,  dim=list(range(self.data_x.dim()))[:-1])
        geo_locations = self.get_geo_locations()
        return {'adjacency': adjacency_matrix, 'mean': data_mean, 'std': data_std, 'geo_loc':geo_locations}
    
def tensor_collate(list_of_xy:List[Tuple[torch.Tensor]]) -> Tuple[torch.Tensor]:
    """ assume the input is a list of (x, y), pack the x's and y's into two tensors
    """
    xs = [torch.as_tensor(xy[0]).unsqueeze(0) for xy in list_of_xy]
    ys = [torch.as_tensor(xy[1]).unsqueeze(0) for xy in list_of_xy]
    
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)

def to_contiguous(data:torch.Tensor, label:torch.Tensor) -> Tuple[torch.Tensor]:
    
    return {"source": data.contiguous(), "target": label[..., 0].contiguous()}

tensor_to_contiguous = lambda list_of_xy: to_contiguous(*tensor_collate(list_of_xy))

def build_trainvaltest_loaders(
    dataset,
    batch_size=32,
    adj_type='doubletransition'
) -> Tuple[Dict[str, DataLoader], Dict[str, torch.Tensor]]:
    
    trainset = NetworkedTimeSeriesDataset(dataset, split='train', compute_metadata=True, adj_type=adj_type)
    valset = NetworkedTimeSeriesDataset(dataset, split='val', adj_type=adj_type)
    testset = NetworkedTimeSeriesDataset(dataset, split='test', adj_type=adj_type)
    metadata = trainset.metadata
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=tensor_to_contiguous)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=tensor_to_contiguous)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=tensor_to_contiguous)
    
    dataloaders = {'train': trainloader, 'val':valloader, 'test':testloader}
    
    return dataloaders, metadata