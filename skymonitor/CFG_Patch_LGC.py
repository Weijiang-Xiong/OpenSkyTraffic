from omegaconf import OmegaConf
from copy import deepcopy
from skytraffic.config import LazyCall as L
from skytraffic.data.datasets import MetrDataset
from torch.utils.data import DataLoader

from skymonitor.patch_lgc import PatchedMVLSTMGCNConv
from skymonitor.simbarca_explore import SimBarcaExplore
from skymonitor.simbarca_explore_evaluation import SimBarcaExploreEvaluator
from skymonitor.augment import RandomGridCoverage, RandomWalkCoverage

from ..config.common.train import train
from ..config.common.optim import AdamW as optimizer
from ..config.common.schedule import scheduler


train.output_dir = "scratch/patch_lgc_simbarca_explore"

dataset = OmegaConf.create()

dataset.train = L(SimBarcaExplore)(
    split="train",
    input_window=30,
    pred_window=30,
    step_size=3,
    grid_size=220,
    norm_tid=False,
    augmentations=L(RandomWalkCoverage)(
        pts_per_step=1, # step size is 3 min, drone data given every 5 sec, so 36 data points per monitoring step
        cvg_num=10,
        empty_value=0.0,
        data_dims=2,
        ),
)

# the created dataset instance will be passed to the config of data loader in the training script
# if we use relative reference to the train set, like ${..train.input_window}$, then, testloader.dataset.input_window
# will still be a relative reference to testloader.train.input_window, which does not exist.
# so we create a copy of the instance to avoid duplicating the config code, more convenient if we want to make changes.
dataset.test = deepcopy( dataset.train )
dataset.test.split = "test"

dataloader = OmegaConf.create()

dataloader.train = L(DataLoader)(
    dataset=None, # will be set in the training script
    batch_size=8,
    shuffle=True,
    collate_fn=None
)

dataloader.test = L(DataLoader)(
    dataset=None, # will be set in the training script
    batch_size=8,
    shuffle=False,
    collate_fn=None
)

model = L(PatchedMVLSTMGCNConv)(
    use_cvg_mask=True,
    use_global=True,
    feature_dim=4,
    d_model=64,
    temp_patching=1,
    global_downsample_factor=1,
    layernorm=True,
    adjacency_hop=5,
    dropout=0.1,
    input_steps=10,
    pred_steps=10,
    num_nodes=1570,
    pred_feat=2,
    data_null_value=float("nan"),
)

evaluator = L(SimBarcaExploreEvaluator)(
    # we assume that evaluator will be a top-level config, and in the same level we have `train`
    save_dir="${train.output_dir}/evaluation",
    collect_pred=["pred"],
    collect_data=["target"],
    ignore_value=0.0,
    eval_speed=False,
    num_repeat=5,
)
