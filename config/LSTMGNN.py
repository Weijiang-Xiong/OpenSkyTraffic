from .common_cfg import train, scheduler, evaluation
from .common_cfg import adam as optimizer

train.test_best_ckpt = False
train.output_dir = "scratch/lstmgnn"
evaluation.evaluator_type = "metrla"

model = {
    "name": "LSTMGNN",
    "device": "cuda",
    "use_global": True,
    "scale_output": True,
    "normalize_input": False,
    "layernorm": True,
    "d_model": 64
}

dataset = {
    "train": {"name": "metrla_train", "adj_type": "doubletransition"},
    "test": {"name": "metrla_test", "adj_type": "${..train.adj_type}"}
}

dataloader = {
    "train": {"batch_size": 32},
    "test": {"batch_size": "${..train.batch_size}"}
}