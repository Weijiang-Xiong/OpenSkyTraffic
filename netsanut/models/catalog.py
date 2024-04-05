import torch 
import torch.nn as nn 
from omegaconf import DictConfig
from netsanut.utils import Registry
from copy import deepcopy

MODEL_CATALOG = Registry("MODEL_CATALOG")

def build_model(cfg: DictConfig) -> nn.Module:
    """ Instantiate a model with the class retrieved from `MODEL_CATALOG` using `cfg.model.name`
    """
    model_cfg = deepcopy(cfg)
    model_architecture, device = model_cfg.pop("name"), model_cfg.pop("device")
    model = MODEL_CATALOG.get(model_architecture)(**model_cfg)
    model.to(torch.device(device))
    return model 
