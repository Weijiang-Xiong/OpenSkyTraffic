import torch

from skytraffic.config import LazyCall as L
from skytraffic.solver import WarmupMultiStepScaler
from .train import train as train_cfg

scheduler = L(torch.optim.lr_scheduler.LambdaLR)(
    optimizer=None, 
    lr_lambda=L(WarmupMultiStepScaler)(
    start=0, 
    end=train_cfg.max_epoch, 
    milestones=[0.7, 0.85], 
    gamma=0.1, 
    warmup=2.0
    )
)
