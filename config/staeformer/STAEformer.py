from omegaconf import OmegaConf
from skytraffic.config import LazyCall as L
from skytraffic.models import STAEformer
from skytraffic.data.datasets import MetrDataset
from torch.utils.data import DataLoader

from ..common.train import train
from ..common.data import metrla as dataset
from ..common.evaluation import metr_evaluator as evaluator
from ..common.optim import AdamW as optimizer
from ..common.schedule import scheduler

train.output_dir = "scratch/metr_staeformer"
train.max_epoch = 20

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

model = L(STAEformer)(
    # arguments purely based on model
    steps_per_day=288,
    input_dim=1,
    output_dim=1,
    input_embedding_dim=24,
    tod_embedding_dim=24,
    dow_embedding_dim=24,
    spatial_embedding_dim=0,
    adaptive_embedding_dim=80,
    feed_forward_dim=256,
    num_heads=4,
    num_layers=3,
    dropout=0.1,
    use_mixed_proj=True,
    add_time_in_day=True,
    add_day_in_week=False,
    loss_ignore_value = float("nan"),
    norm_label_for_loss=True,
    # arguments related to dataset
    input_steps=MetrDataset.input_steps,
    pred_steps=MetrDataset.pred_steps,
    num_nodes=MetrDataset.num_nodes,
    data_null_value=MetrDataset.data_null_value,
    metadata=None,
) 