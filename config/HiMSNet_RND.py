from .common_cfg import train, scheduler, evaluation
from .common_cfg import adam as optimizer
from .HiMSNet import model
from .datasets import simbarca_rnd

train.test_best_ckpt = False
train.output_dir = "scratch/himsnet_rndobsv"
evaluation.evaluator_type = "simbarca"
evaluation.mape_threshold = 1.0
evaluation.ignore_value = float("nan")

dataset = simbarca_rnd
dataloader = {
    "train": {"batch_size": 8},
    "test": {"batch_size": 8}
}