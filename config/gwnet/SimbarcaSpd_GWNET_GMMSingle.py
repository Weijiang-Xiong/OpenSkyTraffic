from .SimbarcaSpd_GWNET_GMM import (
    train, 
    model,
    optimizer,
    scheduler,
    dataloader,
    dataset,
    evaluator,
)

# Override train settings
train.output_dir = "scratch/simbarcaspd_gwnet_gmm_single"

model.anchors = [0.0]
model.sizes = [3.0] 