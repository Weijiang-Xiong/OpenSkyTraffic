from .common_cfg import train, scheduler, evaluation
from .common_cfg import adam as optimizer

train.test_best_ckpt = False
train.output_dir = "scratch/gmmpred_fullinfo"
evaluation.evaluator_type = "simbarcagmm"
evaluation.save_note = "gmmpred"
evaluation.mape_threshold = 1.0
evaluation.ignore_value = float("nan")
evaluation.add_output_seq = ["seg_mixing","seg_means","seg_log_var","reg_mixing","reg_means","reg_log_var"]
model = {
    "name": "GMMPred",
    "device": "cuda",
    "use_drone": True,
    "use_ld": True,
    "use_global": True,
    "scale_output": True,
    "normalize_input": True,
    "layernorm": True,
    "d_model": 64,
    "simple_fillna": False,
    "global_downsample_factor": 1,
    "adjacency_hop": 5,
    "dropout": 0.1,
    "zero_init": True,
}

dataset = {
    "train": {"name": "simbarca_train", "force_reload": False},
    "test": {"name": "simbarca_test", "force_reload": False},
}

dataloader = {
    "train": {"batch_size": 8},
    "test": {"batch_size": 8}
}