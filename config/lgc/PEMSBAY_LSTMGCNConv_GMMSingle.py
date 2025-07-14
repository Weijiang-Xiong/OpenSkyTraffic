from .LSTMGCNConv_GMMSingle import (
    train, 
    model,
    optimizer,
    scheduler,
    dataloader,
    evaluator
)
from ..common.data import pemsbay as dataset
from skytraffic.data.datasets import PEMSBayDataset

train.output_dir = "scratch/pemsbay_lgc_gmm_single"
model.num_nodes = PEMSBayDataset.num_nodes
evaluator.sp_size = 3 # reduce memory usage for GMM evaluation