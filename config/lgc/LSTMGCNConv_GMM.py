from skytraffic.config import LazyCall as L
from skytraffic.models import LSTMGCNConv_GMM
from skytraffic.data.datasets import MetrDataset

from ..common.train import train
from ..common.evaluation import metr_gmm_evaluator as evaluator
from ..common.optim import AdamW as optimizer
from ..common.schedule import scheduler
from .LSTMGCNConv import dataset, dataloader

# Override train settings
train.output_dir = "scratch/metr_lgc_gmm"

model = L(LSTMGCNConv_GMM)(
    # arguments purely based on model
    use_global=True,
    d_model=64,
    global_downsample_factor=1,
    layernorm=True,
    adjacency_hop=1,
    dropout=0.1,
    loss_ignore_value = float("nan"),
    # GMM-specific parameters
    anchors=[-2.0, -1.0, 0.0, 1.0, 2.0],
    sizes=[1.0, 1.0, 1.0, 1.0, 1.0],
    # arguments related to dataset
    input_steps=MetrDataset.input_steps,
    pred_steps=MetrDataset.pred_steps,
    num_nodes=MetrDataset.num_nodes,
    data_null_value=0.0,
    metadata=None,
) 
