""" PEMS-Bay dataset has exactly the same structure as METR-LA, so we can copy most configs from METR-LA
    with very minor changes.
    We only need to change the dataset, output path and number of nodes of the model. 
"""
from .PEMSBAY_GWNET_GMM import (
    train, 
    model,
    optimizer,
    scheduler,
    dataloader,
    dataset,
    evaluator
)

# Override train settings
train.output_dir = "scratch/pemsbay_gwnet_gmm_single"

model.anchors = [0.0]
model.sizes = [3.0] 