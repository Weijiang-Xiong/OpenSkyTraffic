""" PEMS-Bay dataset has exactly the same structure as METR-LA, so we can copy most configs from METR-LA
    with very minor changes.
    We only need to change the dataset, output path and number of nodes of the model. 
"""
from .LSTMGCNConv import (
    train, 
    model,
    optimizer,
    scheduler,
    dataloader,
    evaluator
)
from ..common.data import pemsbay as dataset
from skytraffic.data.datasets import PEMSBayDataset

train.output_dir = "scratch/pemsbay_lgc"
model.num_nodes = PEMSBayDataset.num_nodes
evaluator.data_max = 85.0 # the max speed of PEMS-Bay dataset is 85