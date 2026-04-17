from skytraffic.config import LazyCall as L
from skytraffic.models import ForecastModel, GMMTensorDataNormalizer, STIDGMMNet
from skytraffic.data.datasets import MetrDataset

from ..common.train import train
from ..common.evaluation import metr_gmm_evaluator as evaluator
from ..common.optim import AdamW as optimizer
from ..common.schedule import scheduler
from .STID import dataset, dataloader

# Override train settings
train.output_dir = "scratch/metr_stid_gmm"

model = L(ForecastModel)(
    model=L(STIDGMMNet)(
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
        anchors=[-2.0, -1.0, 0.0, 1.0, 2.0],
        sizes=[1.0, 1.0, 1.0, 1.0, 1.0],
        zero_init=True,
        mcd_estimation=False,
        input_steps=MetrDataset.input_steps,
        pred_steps=MetrDataset.pred_steps,
        num_nodes=MetrDataset.num_nodes,
    ),
    normalizer=L(GMMTensorDataNormalizer)(),
    data_null_value=MetrDataset.data_null_value,
    metadata=None,
) 
