from omegaconf import OmegaConf
from skytraffic.config import LazyCall as L
from skytraffic.models import MTGNN
from skytraffic.data.datasets import MetrDataset
from torch.utils.data import DataLoader

from ..common.train import train
from ..common.data import metrla as dataset
from ..common.evaluation import metr_evaluator as evaluator
from ..common.optim import AdamW as optimizer
from ..common.schedule import scheduler

train.output_dir = "scratch/metr_mtgnn"

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

model = L(MTGNN)(
    # arguments purely based on model
    gcn_true=True,
    buildA_true=True,
    gcn_depth=2,
    dropout=0.3,
    subgraph_size=20,
    node_dim=40,
    dilation_exponential=1,
    conv_channels=32,
    residual_channels=32,
    skip_channels=64,
    end_channels=128,
    layers=3,
    propalpha=0.05,
    tanhalpha=3,
    layer_norm_affline=True,
    use_curriculum_learning=False,
    step_size=2500,
    max_epoch=100,
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