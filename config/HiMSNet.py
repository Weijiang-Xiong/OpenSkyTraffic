from .common_cfg import train, scheduler
from .common_cfg import adam as optimizer

train.model_arch = "HiMSNet"
train.test_best_ckpt = False
train.output_dir = "scratch/himsnet"

model = {
    "use_drone": True,
    "use_ld": True,
    "use_global": True,
}
data = {
    "train": {},
    "test": {}
}