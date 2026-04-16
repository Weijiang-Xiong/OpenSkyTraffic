from skytraffic.config import LazyCall as L
from skytraffic.models import STID_GMM
from skytraffic.data.datasets import MetrDataset

from ..common.train import train
from ..common.evaluation import metr_gmm_evaluator as evaluator
from ..common.optim import AdamW as optimizer
from ..common.schedule import scheduler
from .STID import dataset, dataloader

# Override train settings
train.output_dir = "scratch/metr_stid_gmm"

model = L(STID_GMM)(
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
    # GMM-specific parameters
    anchors=[-2.0, -1.0, 0.0, 1.0, 2.0],
    sizes=[1.0, 1.0, 1.0, 1.0, 1.0],
    zero_init=True,
    mcd_estimation=False,
    # arguments related to dataset
    input_steps=MetrDataset.input_steps,
    pred_steps=MetrDataset.pred_steps,
    num_nodes=MetrDataset.num_nodes,
    data_null_value=MetrDataset.data_null_value,
    metadata=None,
) 
