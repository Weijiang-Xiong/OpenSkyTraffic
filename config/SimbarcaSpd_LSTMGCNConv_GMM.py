from skytraffic.config import LazyCall as L
from skytraffic.models import LSTMGCNConv_GMM

from .common.train import train
from .common.evaluaiton import simbarca_speed_gmm_evaluator as evaluator
from .common.optim import AdamW as optimizer
from .common.schedule import scheduler
from .SimbarcaSpd_LSTMGCNConv import dataset, dataloader

# Override train settings
train.output_dir = "scratch/lgc_gmm_simbarcaspd"

model = L(LSTMGCNConv_GMM)(
    # arguments purely based on model
    use_global=True,
    d_model=64,
    global_downsample_factor=1,
    layernorm=True,
    adjacency_hop=1,
    dropout=0.1,
    loss_ignore_value = float("nan"),
    norm_label_for_loss=True,
    # GMM-specific parameters
    anchors=[-2.0, -1.0, 0.0, 1.0, 2.0],
    sizes=[1.0, 1.0, 1.0, 1.0, 1.0],
    # arguments related to dataset (SimbarcaSpd specific)
    input_steps=10,
    pred_steps=10,
    num_nodes=1570,
    data_null_value=float('nan'),
    metadata=None,
) 