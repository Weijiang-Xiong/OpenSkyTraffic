from skytraffic.config import LazyCall as L
from skytraffic.models import NeTSFormer

from .common.train import train
from .common.evaluaiton import metr_evaluator as evaluator
from .NeTSFormer_prediction import dataset, dataloader

# Override train settings
train.checkpoint = "scratch/prediction/model_final.pth"
train.test_best_ckpt = False
train.max_epoch = 20

# Custom optimizer configuration with groups
optimizer = {
    # metadata about optimizer configuration
    "type": "adam",
    # if there is only one group, then "groups" should be set to `None`
    # and no group-specific hyper-parameters should exists in the config
    "groups": ["det", "sto"],
    # group-specific hyper-parameters, will override common ones
    "sto": {
        "lr": 0.002,
    },
    "det": {
        "lr": 0.000,
    },
    # common hyper-parameters
    "lr": 0.001,
    "weight_decay": 0.0001,
    "betas": (0.9, 0.999)
}

# Custom scheduler configuration with groups  
scheduler = {
    "groups": ["det", "sto"],  # Reference to optimizer groups
    "det": {
        "lr_milestone": [0.7, 0.85],
    },
    "sto": {
        "lr_milestone": [0.7, 0.85],
    },
    "start": 0,
    "end": 20,  # train.max_epoch
    "lr_decrease": 0.1,
    "warmup": 1.0,
}

model = L(NeTSFormer)(
    # model architecture parameters
    in_dim=2,
    hid_dim=64,
    ff_dim=256,
    hist_len=12,
    pred_len=12,
    nhead=2,
    dropout=0.1,
    encoder_layers=2,
    decoder_layers=2,
    time_first=True,
    temp_aggregate="avg",
    # positional encoding parameters
    se_type="learned",  # spatial encoding
    se_init="rand",     # spatial encoding initialization
    te_type="fixed",    # temporal encoding
    te_init="",         # temporal encoding initialization
    # loss parameters
    reduction="mean",
    aleatoric=True,     # Enable aleatoric uncertainty
    exponent=1,
    alpha=1.0,
    ignore_value=0.0,
    temp_causal=False,  # add causal mask to temporal attention of encoder
    # arguments related to dataset/training
    metadata=None,
) 