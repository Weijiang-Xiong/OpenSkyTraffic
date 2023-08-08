from .NeTSFormer_prediction import train, data, model

train.checkpoint = "scratch/prediction/model_final.pth"
train.test_best_ckpt = False
train.max_epoch = 20

model.aleatoric=True

optimizer = {
    # metadata about optimizer configuration
    "type": "adam",
    # if there is only one group, then "groups" should be set to `None`
    # and no group-specific hyper-parameters should exists in the config
    "groups": ["det", "sto"],
    # group-specific hyper-parameters, will override common specific ones
    "sto": {
        "lr": 0.002,
    },
    "det": {
        "lr": 0.000,
    },
    # common hyper-parameters
    "weight_decay": 0.0001,
    "betas": (0.9, 0.999)
}

scheduler = {
    "groups": "${..optimizer.groups}",
    "det":{
        "lr_milestone": [0.7, 0.85],
    },
    "sto":{
        "lr_milestone": [0.7, 0.85],
    },
    "start": 0,
    "end": "${..train.max_epoch}",
    "lr_decrease": 0.1, 
    "warmup": 1.0,
}