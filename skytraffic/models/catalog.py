from copy import deepcopy
from omegaconf import DictConfig

import torch 

from .base import BaseModel
from ..utils.registry import Registry

MODEL_CATALOG = Registry("MODEL_CATALOG")

def build_model(cfg: DictConfig, metadata: dict = None) -> BaseModel:
    """ Instantiate a model with the class retrieved from `MODEL_CATALOG` using `cfg.model.name`
    """
    model_cfg = deepcopy(cfg)
    model_architecture, device = model_cfg.pop("name"), model_cfg.pop("device")
    model:BaseModel = MODEL_CATALOG.get(model_architecture)(**model_cfg, metadata=metadata)
    model.to(torch.device(device))

    return model
