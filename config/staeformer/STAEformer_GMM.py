from skytraffic.config import LazyCall as L
from skytraffic.models import ForecastModel, GMMTensorDataNormalizer, STAEformer_GMM
from skytraffic.data.datasets import MetrDataset

from ..common.train import train
from ..common.evaluation import metr_gmm_evaluator as evaluator
from ..common.optim import AdamW as optimizer
from ..common.schedule import scheduler
from .STAEformer import dataset, dataloader

# Override train settings
train.output_dir = "scratch/metr_staeformer_gmm"
train.max_epoch = 20

model = L(ForecastModel)(
    model=L(STAEformer_GMM)(
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
        add_time_in_day=True,
        add_day_in_week=False,
        anchors=[-2.0, -1.0, 0.0, 1.0, 2.0],
        sizes=[1.0, 1.0, 1.0, 1.0, 1.0],
        zero_init=True,
        mcd_estimation=False,
        input_steps=MetrDataset.input_steps,
        pred_steps=MetrDataset.pred_steps,
        num_nodes=MetrDataset.num_nodes,
        metadata="${..metadata}",
    ),
    normalizer=L(GMMTensorDataNormalizer)(),
    data_null_value=MetrDataset.data_null_value,
    metadata=None,
)
