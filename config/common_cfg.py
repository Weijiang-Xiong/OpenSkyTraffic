train = {
    "max_epoch": 30,
    "output_dir": "scratch/debug", 
    "checkpoint": "",
    "device": "cuda",
    # select model by validation set performance, and test the best model
    # will test the final model if set to False
    "best_metric": None,
    "test_best_ckpt": False, 
    "grad_clip": 3.0,
    "eval_train": False, # run evaluation on train set after each epoch
    "eval_period": 1, # run evaluation every n epochs
    "save_period": 5, # save checkpoint every n epochs
}

adam = {
    "type": "adam",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "betas": (0.9, 0.999)
}

adamw = {
    "type": "adamw",
    "lr": 5e-4,                # Slightly higher than Adam
    "weight_decay": 1e-4,      # Higher weight decay (AdamW handles this better)
    "betas": (0.9, 0.999),     # Standard values
    "eps": 1e-8
}

sgd = {
    "type": "sgd",
    "lr": 0.001,
    "momentum": 0,
    "weight_decay": 0
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

evaluation = {
    "evaluator_type": "",
    "save_dir": "${..train.output_dir}/evaluation",
}