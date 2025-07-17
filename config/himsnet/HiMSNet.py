from omegaconf import OmegaConf
from skytraffic.config import LazyCall as L
from skytraffic.models import HiMSNet
from torch.utils.data import DataLoader

from ..common.train import train
from ..common.data import simbarca_msmt as dataset
from ..common.evaluation import simbarca_evaluator as evaluator
from ..common.optim import Adam as optimizer
from ..common.schedule import scheduler

# Override train settings
train.output_dir = "scratch/simbarca_full_himsnet"

dataloader = OmegaConf.create()

dataloader.train = L(DataLoader)(
    dataset=dataset.train,
    batch_size=8,
    shuffle=True,
    collate_fn=None
)

dataloader.test = L(DataLoader)(
    dataset=dataset.test,
    batch_size=8,
    shuffle=False,
    collate_fn=None
)

model = L(HiMSNet)(
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
    attn_agg=True,
    # arguments related to dataset/training
    metadata=None,
) 