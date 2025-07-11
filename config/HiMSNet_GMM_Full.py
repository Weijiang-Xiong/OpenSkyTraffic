from skytraffic.config import LazyCall as L
from skytraffic.models import HiMSNet_GMM

from .common.train import train
from .common.evaluation import simbarca_gmm_evaluator as evaluator
from .common.optim import Adam as optimizer
from .common.schedule import scheduler
from .HiMSNet import dataset, dataloader

# Override train settings
train.output_dir = "scratch/gmmpred_fullinfo"

# Override evaluator settings for additional output sequences
evaluator.add_output_seq = ["seg_mixing","seg_means","seg_log_var","reg_mixing","reg_means","reg_log_var"]

model = L(HiMSNet_GMM)(
    # arguments purely based on model
    use_drone=True,
    use_ld=True,
    use_global=True,
    scale_output=True,
    normalize_input=True,
    layernorm=True,
    d_model=64,
    simple_fillna=False,
    global_downsample_factor=1,
    adjacency_hop=5,
    dropout=0.1,
    zero_init=True,
    # arguments related to dataset/training
    metadata=None,
) 