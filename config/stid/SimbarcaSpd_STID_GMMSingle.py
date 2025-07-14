from .SimbarcaSpd_STID_GMM import (
    train, 
    model,
    optimizer,
    scheduler,
    dataloader,
    dataset,
    evaluator,
)

# Override train settings
train.output_dir = "scratch/simbarcaspd_stid_gmm_single"

model.anchors = [0.0]
model.sizes = [3.0] 