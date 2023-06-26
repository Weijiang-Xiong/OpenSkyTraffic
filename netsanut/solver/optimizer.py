import torch.nn as nn 
import torch.optim as optim 
from omegaconf import DictConfig, OmegaConf

def build_optimizer(model: nn.Module, cfg: DictConfig) -> optim.Optimizer:
    
    cfg = cfg.copy()
    
    optimizer_type = cfg.pop("type", None)
    config_groups = cfg.pop("groups", None)
    
    # different parameters may have different settings
    if config_groups is not None:
        param_groups = model.get_param_groups()
        config_groups = [g for g in config_groups if g in param_groups.keys()] 
        # pop the group configs from cfg, so the remainders are treated as common configs
        group_specific_config = {g:OmegaConf.to_container(cfg.pop(g)) for g in config_groups}
        for g in config_groups:
            group_specific_config[g]['params'] = param_groups[g] 
            
        optimizer = optim.Adam(
            [group_specific_config[g] for g in config_groups],
            **cfg
        )
    # all parameters share the same set of optimizer settings
    else:
        optimizer = optim.Adam(
            model.parameters(),
            **cfg
        )

    return optimizer