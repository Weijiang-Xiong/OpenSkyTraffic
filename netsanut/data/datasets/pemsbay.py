import pandas as pd
from ..catalog import DATASET_CATALOG
from .metrla import MetrDataset

class PEMSBayDataset(MetrDataset):

    data_root     = 'datasets/pems'
    split_files   = {
        "train": "train.npz",
        "val": "val.npz",
        "test": "test.npz"
    }
    adjacency     = 'pems/adj_mx_bay.pkl',
    geo_locations = 'pems/graph_sensor_locations_bay.csv'

    def __init__(self, split, **args) -> None:
        super().__init__(name='pemsbay', split=split, **args)
        
    def get_geo_locations(self):
        return pd.read_csv(DATASET_CATALOG[self.name]['geo_locations'], header=None)[[2,1]].to_numpy()
    

if __name__.startswith(".pemsbay"):
    DATASET_CATALOG['pemsbay_train'] = lambda **arg: PEMSBayDataset(split='train', **arg)
    DATASET_CATALOG['pemsbay_val'] = lambda **arg: PEMSBayDataset(split='val', **arg)
    DATASET_CATALOG['pemsbay_test'] = lambda **arg: PEMSBayDataset(split='test', **arg)