from skytraffic.config import LazyCall as L
from skytraffic.models import LSTMGCNConv_GMM

from .common.train import train
from .common.evaluaiton import metr_gmm_evaluator as evaluator
from .common.optim import AdamW as optimizer
from .common.schedule import scheduler
from .LSTMGCNConv import dataset, dataloader

# Override train settings
train.test_best_ckpt = False
train.output_dir = "scratch/lgc_gmm_single"

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
    # GMM-specific parameters (single anchor/size)
    anchors=[0.0],
    sizes=[3.0],
    # arguments related to dataset
    input_steps=12,
    pred_steps=12,
    num_nodes=207,
    data_null_value=0.0,
    metadata=None,
) 