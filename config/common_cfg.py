train = {
    "max_epoch": 30,
    "output_dir": "scratch/debug", 
    "checkpoint": "",
    "device": "cuda",
    # select model by validation set performance, and test the best model
    # will test the final model if set to False
    "test_best_ckpt": True, 
    "grad_clip": 3.0,
    "eval_train": True, # run evaluation on train set after each epoch
}

data = {
    "dataset": "metr-la",
    "adj_type": "doubletransition",
    "batch_size": 32,
}

adam = {
    "type": "adam",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "betas": (0.9, 0.999)
}

adamw = {
    "type": "adamw",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "betas": (0.9, 0.999)
}

sgd = {
    "type": "sgd",
    "lr": 1e-2,
    "momentum": 0.9
}

scheduler = {
    "start": 0,
    "end": "${..train.max_epoch}",
    # decrease learning rate when training reaches proportions of train.max_epoch
    "lr_milestone": [0.7, 0.85],
    # multiply the learning rate by lr_decrease at each milestone
    "lr_decrease": 0.1, 
    # gradually increase the learning rate in warmup epochs 
    "warmup": 1.0,
}