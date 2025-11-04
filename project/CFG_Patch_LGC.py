from omegaconf import OmegaConf
from skytraffic.config import LazyCall as L
from skytraffic.data.datasets import MetrDataset
from torch.utils.data import DataLoader

from .patch_lgc import PatchedMVLSTMGCNConv
from .simbarca_explore import SimBarcaExplore
from .simbarca_explore_evaluation import SimBarcaExploreEvaluator
from .augment import RandomGridCoverage

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
    num_unpadded_samples=20,
    grid_size=220,
    allow_shorter_input=True,
    pad_input=True,
    augmentations=L(RandomGridCoverage)(
        input_window=30, 
        step_size=3, 
        num_positions=10, 
        empty_value=0.0
        ),
)
dataset.test = L(SimBarcaExplore)(
    split="test",
    input_window=30,
    pred_window=30,
    step_size=3,
    num_unpadded_samples=20,
    grid_size=220,
    allow_shorter_input=True,
    pad_input=True,
    augmentations=L(RandomGridCoverage)(
        input_window=30, 
        step_size=3, 
        num_positions=10, 
        empty_value=0.0
        )
)

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
    use_global=True,
    feature_dim=3,
    d_model=64,
    temp_patching=3,
    global_downsample_factor=1,
    layernorm=True,
    adjacency_hop=1,
    dropout=0.1,
    loss_ignore_value = float("nan"),
    norm_label_for_loss = True,
    input_steps=360,
    pred_steps=10,
    num_nodes=1570,
    pred_feat=2,
    data_null_value=0.0,
)

evaluator = L(SimBarcaExploreEvaluator)(
    # we assume that evaluator will be a top-level config, and in the same level we have `train`
    save_dir="${train.output_dir}/evaluation",
    visualize=False,
    collect_pred=["pred"],
    collect_data=["target"],
    ignore_value=0.0,
    eval_speed=False,
)