from .NeTSFormer_stable import train, data, model

train.test_best_ckpt = False
train.max_epoch = 25
train.milestone = 20
train.milestone_cfg = {
    "model": {
        "loss": {
            "reduction": "mean",
            "aleatoric": True, 
            "exponent": 1,
            "alpha": 1.0,
            "ignore_value": 0.0
        }
    }
}

optimizer = {
    # metadata about optimizer configuration
    "type": "adam",
    # if there is only one group, then "groups" should be set to `None`
    # and no group-specific hyper-parameters should exists in the config
    "groups": ["det", "sto"],
    # group-specific hyper-parameters
    "det": {
        "lr": 0.001,
    },
    "sto": {
        "lr": 0.001,
    },
    # common hyper-parameters
    # all of them will be unpacked to initialize the optimizer
    "weight_decay": 0.0001,
}

scheduler = {
    "groups": "${..optimizer.groups}",
    "det":{
        "start": 0,
        "end": "${...train.milestone}",
        "lr_milestone": [0.7, 0.85],
    },
    "sto":{
        "start": "${...train.milestone}",
        "end": "${...train.max_epoch}",
        "lr_milestone": [0.7, 0.8],
    },
    "lr_decrease": 0.1, 
    "warmup": 1.0,
}