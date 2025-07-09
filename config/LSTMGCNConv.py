from .common_cfg import train, scheduler, evaluation
from .common_cfg import adamw as optimizer
from .datasets import metrla as dataset 

train.test_best_ckpt = False
train.output_dir = "scratch/lgc"
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
    "adjacency_hop": 1,
    "dropout": 0.1,
    "norm_label_for_loss": True,
    "input_steps": 12,
    "pred_steps": 12,
    "num_nodes": 207,
}

dataloader = {
    "train": {"batch_size": 32},
    "test": {"batch_size": "${..train.batch_size}"}
}