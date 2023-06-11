from omegaconf import OmegaConf

train = OmegaConf.create({
    "output_dir": "scratch/test", 
    "checkpoint": "",
    "device": "cuda"
})

data = OmegaConf.create({
    "dataset": "metr-la",
    "adj_type": "doubletransition",
    "batch_size": 32,
})

optimizer = OmegaConf.create({
    "max_epoch": 30,
    "learning_rate": 0.001,
    "lr_milestone": [0.7, 0.85],
    "lr_decrease": 0.1,
    "dropout": 0.1,
    "weight_decay": 0.0001,
    "grad_clip": 3.0
})