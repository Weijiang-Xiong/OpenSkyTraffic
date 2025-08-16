""" PEMS-Bay dataset has exactly the same structure as METR-LA, so we can copy most configs from METR-LA
    with very minor changes.
    We only need to change the dataset, output path and number of nodes of the model. 
"""
from .PEMSBAY_MTGNN_GMM import (
    train, 
    model,
    optimizer,
    scheduler,
    dataloader,
    dataset,
    evaluator
)

# Override train settings
train.output_dir = "scratch/pemsbay_mtgnn_gmm_single"
evaluator.data_max = 85.0 # the max speed of PEMS-Bay dataset is 85

model.anchors = [0.0]
model.sizes = [3.0] 