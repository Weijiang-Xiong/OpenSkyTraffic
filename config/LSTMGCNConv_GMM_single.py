from .common_cfg import train, scheduler, evaluation
from .common_cfg import adam as optimizer
from .LSTMGCNConv import dataset, dataloader, model

train.test_best_ckpt = False
train.output_dir = "scratch/lstmgnngmm"
evaluation.evaluator_type = "metrlagmm"

model.name = "LSTMGCNConv_GMM"
model.anchors = [0.0]
model.sizes = [3.0]