from omegaconf import OmegaConf
from skytraffic.config import LazyCall as L
from skytraffic.models import LSTMGCNConv
from torch.utils.data import DataLoader

from .common.train import train
from .common.data import simbarcaspd as dataset
from .common.evaluaiton import simbarca_speed_evaluator as evaluator
from .common.optim import AdamW as optimizer
from .common.schedule import scheduler

# Override train settings
train.output_dir = "scratch/lgc_simbarcaspd"

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

model = L(LSTMGCNConv)(
    # arguments purely based on model
    use_global=True,
    d_model=64,
    global_downsample_factor=1,
    layernorm=True,
    adjacency_hop=1,
    dropout=0.1,
    loss_ignore_value = float("nan"),
    norm_label_for_loss=True,
    # arguments related to dataset (SimbarcaSpd specific)
    input_steps=10,
    pred_steps=10,
    num_nodes=1570,
    data_null_value=float('nan'),
    metadata=None,
) 