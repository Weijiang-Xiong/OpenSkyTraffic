from .common_cfg import train, scheduler, evaluation
from .common_cfg import adamw as optimizer
from .LSTMGCNConv import dataset, dataloader, model

train.test_best_ckpt = False
train.output_dir = "scratch/lgc_gmm"
evaluation.evaluator_type = "metrlagmm"

model.name = "LSTMGCNConv_GMM"
model.anchors = [-2.0, -1.0, 0.0, 1.0, 2.0]
model.sizes = [1.0, 1.0, 1.0, 1.0, 1.0]