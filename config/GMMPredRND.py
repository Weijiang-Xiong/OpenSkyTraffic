from .common_cfg import train, scheduler, evaluation
from .common_cfg import adam as optimizer
from .GMMPredFull import model

train.test_best_ckpt = False
train.output_dir = "scratch/gmmpred_rndobsv"
evaluation.evaluator_type = "simbarcagmm"
evaluation.save_res = False
evaluation.save_note = "gmmpred"
evaluation.mape_threshold = 1.0
evaluation.ignore_value = float("nan")
evaluation.add_output_seq = ["seg_mixing","seg_means","seg_log_var","reg_mixing","reg_means","reg_log_var"]


dataset = {
    "train": {"name": "simbarca_rnd_train", "force_reload": False, "use_clean_data": False},
    "test": {"name": "simbarca_rnd_test", "force_reload": False, "use_clean_data": "${..train.use_clean_data}"},
}

dataloader = {
    "train": {"batch_size": 8},
    "test": {"batch_size": 8}
}