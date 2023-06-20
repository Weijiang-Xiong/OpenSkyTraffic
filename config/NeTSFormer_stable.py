from .common_cfg import train, data, optimizer, scheduler

train.model_arch = "netsformer"
model = {
    # these are information about the model
    "name": "netsformer",
    "device": "cuda",
    # these are model parameters
    "in_dim": 2, 
    "hid_dim": 64,
    "ff_dim": 256,
    "out_dim": 12,
    "nhead": 2,
    "dropout": 0.1,
    "encoder_layers":2,
    "decoder_layers": 2,
    "time_first": True,
    # these are related to loss
    "reduction": "mean",
    "aleatoric": False, 
    "exponent": 1,
    "alpha": 1.0,
    "ignore_value": 0.0
}
