from .SimbarcaSpd_LSTMGCNConv_GMM import (
    train, 
    model,
    optimizer,
    scheduler,
    dataset,
    evaluator,
    dataloader,
)


train.output_dir = "scratch/simbarcaspd_lgc_gmm_single"

model.anchors = [0.0]
model.sizes = [3.0]