from .common_cfg import train, scheduler
from .common_cfg import adam as optimizer

train.model_arch = "netsformer"
model = {}
data = {
    "dataset": "simbarca",
    "adj_type": "doubletransition",
    "batch_size": 8,
}