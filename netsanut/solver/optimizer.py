import torch.nn as nn 
import torch.optim as optim 
from omegaconf import DictConfig, OmegaConf

def build_optimizer(model: nn.Module, cfg: DictConfig) -> optim.Optimizer:
    
    cfg = cfg.copy()
    
    optimizer_type = cfg.pop("type", None)
    config_groups = cfg.pop("groups", None)
    
    # parse the config by parameter group
    # different parameters may have different settings
    if config_groups is not None:
        param_groups = model.get_param_groups()
        config_groups = [g for g in config_groups if g in param_groups.keys()] 
        # pop the group configs from cfg, use an empty dict if no specific config is given
        group_specific_config = {g:OmegaConf.to_container(cfg.pop(g, OmegaConf.create({}))) for g in config_groups}
        for g in config_groups:
            group_specific_config[g]['params'] = param_groups[g] 
        
        params = [group_specific_config[g] for g in config_groups]
        
    # all parameters share the same set of optimizer settings
    else:
        params = model.parameters()

    # initialize the optimizer
    match optimizer_type:
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
                momentum=cfg.momentum
            )
    
    return optimizer