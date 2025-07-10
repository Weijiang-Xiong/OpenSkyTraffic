from omegaconf import OmegaConf
from skytraffic.config import LazyCall as L
from skytraffic.models import NeTSFormer
from torch.utils.data import DataLoader

from .common.train import train
from .common.data import metrla as dataset
from .common.evaluaiton import metr_evaluator as evaluator
from .common.optim import Adam as optimizer
from .common.schedule import scheduler

dataloader = OmegaConf.create()

dataloader.train = L(DataLoader)(
    dataset=dataset.train,
    batch_size=32,
    shuffle=True,
    collate_fn=None
)

dataloader.test = L(DataLoader)(
    dataset=dataset.test,
    batch_size="${..train.batch_size}",
    shuffle=False,
    collate_fn=None
)

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
    aleatoric=False,
    exponent=1,
    alpha=1.0,
    ignore_value=0.0,
    temp_causal=False,  # add causal mask to temporal attention of encoder
    # arguments related to dataset/training
    metadata=None,
) 