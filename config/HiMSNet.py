from .common_cfg import train, scheduler, evaluation
from .common_cfg import adam as optimizer

train.test_best_ckpt = False
train.output_dir = "scratch/himsnet"
evaluation.evaluator_type = "simbarca"

model = {
    "name": "HiMSNet",
    "device": "cuda",
    "use_drone": True,
    "use_ld": True,
    "use_global": True,
    "scale_output": True,
    "normalize_input": False,
    "layernorm": True,
    "d_model": 64
}

dataset = {
    "train": {"name": "simbarca_train", "force_reload": False},
    "test": {"name": "simbarca_test", "force_reload": False},
}

dataloader = {
    "train": {"batch_size": 8},
    "test": {"batch_size": 8}
}