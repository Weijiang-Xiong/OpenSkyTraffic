from NeTSFormer_stable import trian, data, optimizer, model

train = {
    "max_epoch": 30,
    "output_dir": "scratch/test", 
    "checkpoint": "",
    "device": "cuda"
}

data = {
    "dataset": "metr-la",
    "adj_type": "doubletransition",
    "batch_size": 32,
}

optimizer = {
    "learning_rate": 0.001,
    "dropout": 0.1,
    "weight_decay": 0.0001,
    "grad_clip": 3.0
}

scheduler = {
    "lr_milestone": [0.7, 0.85],
    "lr_decrease": 0.1,
}

model = {
    # these are information about the model
    "name": "netsformer",
    "device": "cuda",
} 