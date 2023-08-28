from .common_cfg import train, data, scheduler
from .common_cfg import adam as optimizer

train.model_arch = "netsformer"
model = {
    # these are information about the model
    "name": "netsformer",
    "device": "cuda",
    # these are model parameters
    "in_dim": 2, 
    "hid_dim": 64,
    "ff_dim": 256,
    "hist_len": 12,
    "pred_len": 12,
    "nhead": 2,
    "dropout": 0.1,
    "encoder_layers":2,
    "decoder_layers": 2,
    "time_first": True,
    "temp_aggregate": "avg",
    # type of positional encoding, "learned", "fixed" (not learned) or "None" (no encoding)
    "se_type": "learned", # spatial encoding
    "se_init": "rand", # initialization method of spatial encoding, "rand" for random init, "zero" for zero init
    "te_type": "fixed", # temporal encoding
    "te_init": "",
    # these are related to loss
    "reduction": "mean",
    "aleatoric": False, 
    "exponent": 1,
    "alpha": 1.0,
    "ignore_value": 0.0,
    "temp_causal": False # add causal mask to temporal attention of encoder
}
