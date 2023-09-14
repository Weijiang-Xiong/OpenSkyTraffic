import torch
import numpy as np

def _data_loader():
    
    while True:
        yield        

# a data point records (time, value, sensor_type, location) of an observation
data_point = (10, 57, 1, 100)
# a segment contains everything happened in 5 mins
segment = []
# a sample contains 12 segments (one hour) 
sample = [] 
# a batch contains a few samples 







