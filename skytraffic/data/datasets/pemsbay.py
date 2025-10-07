import pandas as pd

from .metrla import MetrDataset

class PEMSBayDataset(MetrDataset):

    data_root     = 'datasets/pems'
    split_files   = {
        "train": "train.npz",
        "val": "val.npz",
        "test": "test.npz"
    }
    adjacency     = 'adj_mx_bay.pkl'
    geo_locations = 'graph_sensor_locations_bay.csv'
    num_nodes = 325

    def __init__(self, split, **args) -> None:
        super().__init__(split=split, **args)
        
    def get_geo_locations(self):
        return pd.read_csv("{}/{}".format(self.data_root, self.geo_locations), header=None)[[2,1]].to_numpy()
    

