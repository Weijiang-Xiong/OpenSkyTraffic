from .common_cfg import train, scheduler
from .common_cfg import adam as optimizer

train.test_best_ckpt = False
train.output_dir = "scratch/himsnet"

model = {
    "name": "himsnet",
    "device": "cuda",
    "use_drone": True,
    "use_ld": True,
    "use_global": True,
    "scale_output": True,
    "normalize_input": False,
    "layernorm": True,
    "d_model": 64
}
data = {
    "train": {"batch_size": 8},
    "test": {"batch_size": 8}
}