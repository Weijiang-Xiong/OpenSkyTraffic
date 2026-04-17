from skytraffic.config import LazyCall as L
from skytraffic.models import ForecastModel, GMMTensorDataNormalizer, MTGNN_GMM
from skytraffic.data.datasets import MetrDataset

from ..common.train import train
from ..common.evaluation import metr_gmm_evaluator as evaluator
from ..common.optim import AdamW as optimizer
from ..common.schedule import scheduler
from .MTGNN import dataset, dataloader

# Override train settings
train.output_dir = "scratch/metr_mtgnn_gmm"

model = L(ForecastModel)(
    model=L(MTGNN_GMM)(
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
