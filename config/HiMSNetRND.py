from .common_cfg import train, scheduler, evaluation
from .common_cfg import adam as optimizer
from .HiMSNet import model

train.test_best_ckpt = False
train.output_dir = "scratch/himsnet_rndobsv"
evaluation.evaluator_type = "simbarca"
evaluation.save_res = False
evaluation.save_note = "example"
evaluation.mape_threshold = 1.0
evaluation.ignore_value = float("nan")

dataset = {
    "train": {"name": "simbarca_rnd_train", "force_reload": False, "use_clean_data": False, "filter_short": None},
    "test": {"name": "simbarca_rnd_test", "force_reload": False, "use_clean_data": "${..train.use_clean_data}", "filter_short":"${..train.filter_short}"},
}

dataloader = {
    "train": {"batch_size": 8},
    "test": {"batch_size": 8}
}