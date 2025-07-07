from .common_cfg import train, scheduler, evaluation
from .common_cfg import adamw as optimizer

train.test_best_ckpt = False
train.output_dir = "scratch/lstmgcnconv"
evaluation.evaluator_type = "metrla"

model = {
    "name": "LSTMGCNConv",
    "device": "cuda",
    "use_global": True,
    "layernorm": True,
    "d_model": 64,
    "global_downsample_factor": 1,
    "data_null_value": 0.0,
    "loss_ignore_value": float("nan"),
    "adjacency_hop": 1
}

dataset = {
    "train": {"name": "metrla_train"},
    "test": {"name": "metrla_test"}
}

dataloader = {
    "train": {"batch_size": 32},
    "test": {"batch_size": "${..train.batch_size}"}
}