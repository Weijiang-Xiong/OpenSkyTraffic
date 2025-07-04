from .common_cfg import train, scheduler, evaluation
from .common_cfg import adam as optimizer
from .datasets import simbarca_msmt

train.test_best_ckpt = False
train.output_dir = "scratch/himsnet"
evaluation.evaluator_type = "simbarca"
evaluation.mape_threshold = 1.0
evaluation.ignore_value = float("nan")

model = {
    "name": "HiMSNet",
    "device": "${train.device}",
    "use_drone": True,
    "use_ld": True,
    "use_global": True,
    "scale_output": True,
    "normalize_input": True, # this will make the results more stable and better
    "layernorm": True,
    "d_model": 64,
    "simple_fillna": False,
    "global_downsample_factor": 1,
    "adjacency_hop": 5,
    "attn_agg": True,
}

dataset = simbarca_msmt
dataloader = {
    "train": {"batch_size": 8},
    "test": {"batch_size": 8}
}