import pickle
import logging
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np 

import torch 
from torch.utils.data import Dataset

from .adjacency import calculate_scaled_laplacian, calculate_normalized_laplacian, sym_adj, asym_adj

logger = logging.getLogger('default')

class MetrDataset(Dataset):
    """ a dataset class for time series on a bunch of nodes (e.g., traffic 
    flow in a city)
    """
    data_root = 'datasets/metr'
    split_files = {
        "train": "train.npz",
        "val": "val.npz",
        "test": "test.npz"
    }
    adjacency = 'adj_mx_metr.pkl'
    geo_locations = 'graph_sensor_locations.csv'
    data_dim = 0
    data_null_value = 0.0
    input_steps = 12
    pred_steps = 12
    num_nodes = 207
    
    def __init__(self, split='train', adj_type='doubletransition') -> None:
        super().__init__()
        self.split = split
        data = np.load("{}/{}".format(self.data_root, self.split_files[self.split]))
        # data assumed to have (N, T, M, C) shape
        self.data_x = torch.as_tensor(data['x'], dtype=torch.float32)
        self.data_y = torch.as_tensor(data['y'], dtype=torch.float32)
        
        self.metadata = self.compute_metadata(adj_type)

    def __len__(self) -> int:
        return int(self.data_x.size(0))
    
    def __getitem__(self, index: int):
        return self.data_x[index], self.data_y[index]
    
    def get_geo_locations(self):
        return pd.read_csv("{}/{}".format(self.data_root, self.geo_locations))[['longitude', "latitude"]].to_numpy()


    def compute_metadata(self, adj_type):
        sensor_ids, sensor_id_to_ind, adj = self.load_adjacency(
            "{}/{}".format(self.data_root, self.adjacency), 
            adj_type
        )
        adjacency_matrix: List[torch.Tensor] = [torch.as_tensor(A) for A in adj]
        # separately calculate the mean and std for each channel
        # although it doesn't make sense to do this for time, we can handle it later
        data_values = self.data_x[..., self.data_dim]
        data_mean = torch.mean(data_values[data_values != self.data_null_value])
        data_std  = torch.std(data_values[data_values != self.data_null_value])
        geo_locations = self.get_geo_locations()
        return {'adjacency': adjacency_matrix, 
                'mean': data_mean, 
                'std': data_std, 
                'data_dim': self.data_dim, 
                'geo_loc':geo_locations,
                'data_null_value': self.data_null_value,
                'input_steps': self.input_steps,
                'pred_steps': self.pred_steps,
                'num_nodes': self.num_nodes
        }

    @staticmethod
    def load_pickle(pickle_file):
        try:
            with open(pickle_file, 'rb') as f:
                pickle_data = pickle.load(f)
        except UnicodeDecodeError:
            with open(pickle_file, 'rb') as f:
                pickle_data = pickle.load(f, encoding='latin1')
        except Exception as e:
            print('Unable to load data ', pickle_file, ':', e)
            raise
        return pickle_data

    # NOTE it's very important to add type hints here. 
    # When writing codes, the language server will try to infer the type of the returned values, 
    # and it's a lot of work for scipy functions because scipy itself doesn't contain good type hints. 
    @classmethod
    def load_adjacency(cls, pkl_filename, adjtype) -> Tuple[List, Dict, List]:
        # sensor_ids, sensor_id_to_ind, adj_mx = self.load_pickle(pkl_filename)
        sensor_ids, sensor_id_to_ind, adj_mx = cls.load_pickle(pkl_filename)
        if adjtype == "scalap":
            adj = [calculate_scaled_laplacian(adj_mx)]
        elif adjtype == "normlap":
            adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
        elif adjtype == "symnadj":
            adj = [sym_adj(adj_mx)]
        elif adjtype == "transition":
            adj = [asym_adj(adj_mx)]
        elif adjtype == "doubletransition":
            adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
        elif adjtype == "identity":
            adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
        else:
            error = 0
            assert error, "adj type not defined"
        return sensor_ids, sensor_id_to_ind, adj

    def collate_fn(self, batch:List[Tuple[torch.Tensor]]):
        """ assume the input is a list of (x, y), pack the x's and y's into two tensors
        """
        xs = [torch.as_tensor(xy[0]).unsqueeze(0) for xy in batch]
        ys = [torch.as_tensor(xy[1]).unsqueeze(0) for xy in batch]
        
        xs, ys = torch.cat(xs, dim=0), torch.cat(ys, dim=0)
        
        return {"source": xs.contiguous(), "target": ys[..., 0].contiguous()}
    
