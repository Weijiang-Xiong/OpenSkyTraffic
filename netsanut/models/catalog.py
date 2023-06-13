import torch 
import torch.nn as nn 
from .ttnet import TTNet
from omegaconf import DictConfig

MODEL_CATALOG = dict()
MODEL_CATALOG['lstm_tf'] = TTNet

def build_model(cfg: DictConfig) -> nn.Module:
    """ Instantiate a model with the class retrieved from `MODEL_CATALOG` using `cfg.model.name`
    """
    model_architecture = cfg.name
    model = MODEL_CATALOG.get(model_architecture)(**cfg)
    model.to(torch.device(cfg.device))
    return model 
