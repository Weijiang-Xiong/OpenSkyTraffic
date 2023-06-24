import torch.nn as nn 
import torch.optim as optim 
from omegaconf import DictConfig, OmegaConf

def build_optimizer(model: nn.Module, cfg: DictConfig) -> optim.Optimizer:
    
    OmegaConf.resolve(cfg) # replace relative interpolations with actual value. 
    params = model.parameters()
    
    optimizer = optim.Adam(
        params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )
    return optimizer