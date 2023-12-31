import json
import numpy as np

import torch
from torch.utils.data import Dataset

class SimBarca(Dataset):
    
    def __init__(self):
        pass 
    
    def __getitem__(self, index):
        pass 

    def __len__(self):
        pass
    
    
def visualize_data():
    pass


if __name__ == '__main__':
    data_file = 'datasets/simbarca/session_000/vehicle_dist_time.json'
    data = json.load(open(data_file, 'r'))
    
