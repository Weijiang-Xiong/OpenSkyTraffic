train = {
    "max_epoch": 30,
    "output_dir": "scratch/debug", 
    "checkpoint": "",
    "device": "cuda",
    # select model by validation set performance, and test the best model
    # will test the final model if set to False
    "test_best_ckpt": True, 
}

data = {
    "dataset": "metr-la",
    "adj_type": "doubletransition",
    "batch_size": 32,
}

optimizer = {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "grad_clip": 3.0
}

scheduler = {
    # decrease learning rate when training reaches proportions of train.max_epoch
    "lr_milestone": [0.7, 0.85],
    # multiply the learning rate by lr_decrease at each milestone
    "lr_decrease": 0.1, 
    # gradually increase the learning rate in warmup epochs 
    "warmup": 1.0,
}