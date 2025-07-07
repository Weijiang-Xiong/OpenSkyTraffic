simbarca_msmt = {
    "train": {"name": "simbarca_msmt_train", "input_window": 30, "pred_window": 30, "step_size": 3, "sample_per_session": 20},
    "test": {"name": "simbarca_msmt_test", 
             "input_window": "${..train.input_window}", 
             "pred_window": "${..train.pred_window}", 
             "step_size": "${..train.step_size}", 
             "sample_per_session": "${..train.sample_per_session}"},
}

randon_observation_params = {
    "ld_cvg": 0.1, 
    "drone_cvg": 0.1, 
    "reinit_pos": False, 
    "mask_seed": 42, 
    "use_clean_data": False, 
    "noise_seed": 114514, 
    "drone_noise": 0.05, 
    "ld_noise": 0.15
}
simbarca_rnd = {
    "train": {"name": "simbarca_rnd_train", "input_window": 30, "pred_window": 30, "step_size": 3, "sample_per_session": 20},
    "test": {"name": "simbarca_rnd_test", 
             "input_window": "${..train.input_window}", 
             "pred_window": "${..train.pred_window}", 
             "step_size": "${..train.step_size}", 
             "sample_per_session": "${..train.sample_per_session}"},
}
simbarca_rnd['train'].update(randon_observation_params)
simbarca_rnd['test'].update(randon_observation_params)

simbarcaspd = {
    "train": {"name": "simbarcaspd_train"},
    "test": {"name": "simbarcaspd_test"},
}