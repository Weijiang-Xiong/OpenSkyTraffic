from omegaconf import OmegaConf
from skytraffic.config import LazyCall as L
from skytraffic.models import ForecastModel, LSTMGCNConv, TensorDataNormalizer
from skytraffic.data.datasets import MetrDataset
from torch.utils.data import DataLoader

from ..common.train import train
from ..common.data import metrla as dataset
from ..common.evaluation import metr_evaluator as evaluator
from ..common.optim import AdamW as optimizer
from ..common.schedule import scheduler

train.output_dir = "scratch/metr_lgc"

dataloader = OmegaConf.create()

dataloader.train = L(DataLoader)(
    dataset=dataset.train,
    batch_size=32,
    shuffle=True,
    collate_fn=None
)

dataloader.test = L(DataLoader)(
    dataset=dataset.test,
    batch_size="${..train.batch_size}",
    shuffle=False,
    collate_fn=None
)

model = L(ForecastModel)(
    model=L(LSTMGCNConv)(
        use_global=True,
        d_model=64,
        global_downsample_factor=1,
        layernorm=True,
        assume_clean_input=True,
        adjacency_hop=1,
        dropout=0.1,
        input_steps=MetrDataset.input_steps,
        pred_steps=MetrDataset.pred_steps,
        num_nodes=MetrDataset.num_nodes,
        metadata="${..metadata}",
    ),
    normalizer=L(TensorDataNormalizer)(),
    data_null_value=MetrDataset.data_null_value,
    metadata=None,
)
