from .common_cfg import train, scheduler
from .common_cfg import adam as optimizer

train.model_arch = "FuseNet"
model = dict()
data = {
    "train": {},
    "test": {}
}