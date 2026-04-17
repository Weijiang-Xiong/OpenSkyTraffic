from .SimbarcaSpd_STAEformer_GMM import (
    train, 
    model,
    optimizer,
    scheduler,
    dataloader,
    dataset,
    evaluator,
)

# Override train settings
train.max_epoch = 20
train.output_dir = "scratch/simbarcaspd_staeformer_gmm_single"
dataloader.train.batch_size = 8
dataloader.test.batch_size = 8
model.model.anchors = [0.0]
model.model.sizes = [3.0]
