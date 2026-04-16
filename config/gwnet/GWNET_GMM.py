from skytraffic.config import LazyCall as L
from skytraffic.models import GWNET_GMM
from skytraffic.data.datasets import MetrDataset

from ..common.train import train
from ..common.evaluation import metr_gmm_evaluator as evaluator
from ..common.optim import AdamW as optimizer
from ..common.schedule import scheduler
from .GWNET import dataset, dataloader

# Override train settings
train.output_dir = "scratch/metr_gwnet_gmm"

model = L(GWNET_GMM)(
    # arguments purely based on model
    dropout=0.3,
    blocks=4,
    layers=2,
    gcn_bool=True,
    addaptadj=True,
    randomadj=True,
    aptonly=True,
    kernel_size=2,
    nhid=32,
    residual_channels=None,
    dilation_channels=None,
    skip_channels=None,
    end_channels=None,
    apt_layer=True,
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
