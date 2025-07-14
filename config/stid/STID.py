from omegaconf import OmegaConf
from skytraffic.config import LazyCall as L
from skytraffic.models import STID
from skytraffic.data.datasets import MetrDataset
from torch.utils.data import DataLoader

from ..common.train import train
from ..common.data import metrla as dataset
from ..common.evaluation import metr_evaluator as evaluator
from ..common.optim import AdamW as optimizer
from ..common.schedule import scheduler

train.output_dir = "scratch/metr_stid"

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

model = L(STID)(
    # arguments purely based on model
    time_intervals=300,
    num_block=2,
    time_series_emb_dim=64,
    spatial_emb_dim=8,
    temp_dim_tid=8,
    temp_dim_diw=8,
    if_spatial=True,
    if_time_in_day=True,
    if_day_in_week=False,
    feature_dim=2,
    output_dim=1,
    loss_ignore_value = float("nan"),
    norm_label_for_loss=True,
    # arguments related to dataset
    input_steps=MetrDataset.input_steps,
    pred_steps=MetrDataset.pred_steps,
    num_nodes=MetrDataset.num_nodes,
    data_null_value=MetrDataset.data_null_value,
    metadata=None,
) 