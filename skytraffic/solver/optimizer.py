import torch.nn as nn 
import torch.optim as optim 

from typing import List, Dict, Any
from collections import defaultdict

from omegaconf import DictConfig

def build_optimizer(model: nn.Module, cfg: DictConfig) -> optim.Optimizer:
    
    """
    Build an optimizer from a model and a configuration. 
    Specifically, the `type` attribute of `cfg` indicates the optimizer type, and the `overrides` attribute of `cfg` indicates the optimizer hyperparameters to override.
    
    One can specify an override as like `cfg.overrides = {"model.encoder": {"lr": 0.01} }`, and then this will replace the default `lr` of the whole model.

    See https://docs.pytorch.org/docs/stable/optim.html for how to specify parameter-specific optimizer settings.
    
    Note that having too many hyperparam groups will make the optimizer slower, so it is recommended to only indicate the parameters that need to be overridden.

    Args:
        model (nn.Module): The model to optimize.
        cfg (DictConfig): The configuration for the optimizer.

    Returns:
        optim.Optimizer: The optimizer.
    """

    cfg = cfg.copy()
    
    # optimizer_type = cfg.pop("type", None)
    # config_groups = cfg.pop("groups", None)
    
    # parse the config by parameter group
    # different parameters may have different settings
    # if config_groups is not None:
    #     param_groups = model.get_param_groups()
    #     config_groups = [g for g in config_groups if g in param_groups.keys()] 
    #     # pop the group configs from cfg, use an empty dict if no specific config is given
    #     group_specific_config = {g:OmegaConf.to_container(cfg.pop(g, OmegaConf.create({}))) for g in config_groups}
    #     for g in config_groups:
    #         group_specific_config[g]['params'] = param_groups[g] 
        
    #     params = [group_specific_config[g] for g in config_groups]
        
    # # all parameters share the same set of optimizer settings
    # else:
    #     params = model.parameters()

    params: List[Dict[str, Any]] = []
    if cfg.get("overrides", None) is None:
        params = model.parameters()
    else:
        params_to_override: Dict[str, List[nn.Parameter]] = defaultdict(list)
        params_using_default: List[nn.Parameter] = []
        overrides: Dict[str, Dict[str, float]] = cfg.overrides

        for module_name, module in model.named_children():
            for _, value in module.named_parameters(recurse=True):
                if not value.requires_grad:
                    continue
                if module_name in overrides:
                    params_to_override[module_name].append(value)
                else:
                    params_using_default.append(value)

        for module_name, module_params in params_to_override.items():
            params.append({"params": module_params, **overrides[module_name]})

        if len(params_using_default) > 0:
            params.append({"params": params_using_default})

    # initialize the optimizer
    match cfg.type:
        case 'adam':
            optimizer = optim.Adam(
                params,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                betas=cfg.betas,
            )
        case 'adamw':
            optimizer = optim.AdamW(
                params,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                betas=cfg.betas,
            )
        case 'sgd':
            optimizer = optim.SGD(
                params,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                momentum=cfg.momentum
            )
    
    return optimizer