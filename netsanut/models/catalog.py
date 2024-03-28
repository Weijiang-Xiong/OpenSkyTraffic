import torch 
import torch.nn as nn 
from omegaconf import DictConfig
from netsanut.utils import Registry

MODEL_CATALOG = Registry("MODEL_CATALOG")

def build_model(cfg: DictConfig) -> nn.Module:
    """ Instantiate a model with the class retrieved from `MODEL_CATALOG` using `cfg.model.name`
    """
    model_architecture = cfg.name
    model = MODEL_CATALOG.get(model_architecture)(**cfg)
    model.to(torch.device(cfg.device))
    return model 
