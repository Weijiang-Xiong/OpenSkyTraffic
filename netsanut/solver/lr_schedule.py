import torch.optim as optim

from typing import Any
from collections import Counter
from bisect import bisect_right

from omegaconf import OmegaConf, DictConfig
class WarmupMultiStepScaler:
    
    def __init__(self, start:int, end:int, milestones:list, gamma:float=0.1, warmup:float=None) -> None:
        
        self.start = start
        self.end = end
        self.milestones = Counter(milestones)
        self.gamma = gamma
        
        assert warmup is None or isinstance(warmup, (int, float))
        self.warmup = warmup
        
    def to_iteration_based(self, iter_per_epoch:int):
        
        assert isinstance(iter_per_epoch, (int))
        self.iteration_based = True
        self.iter_per_epoch = iter_per_epoch
        
        self.start *= iter_per_epoch
        self.end *= iter_per_epoch
        self.milestones = Counter({int(k*iter_per_epoch): v for k, v in self.milestones.items()})
        if self.warmup:
            self.warmup *= iter_per_epoch
        
    def __call__(self, last_step) -> Any:
        
        if last_step < self.start or last_step >= self.end:
            return 0.0
        
        milestones = sorted(self.milestones.elements())
        lr_scale = self.gamma ** bisect_right(milestones, last_step)
        
        if self.warmup:
            # at least 0.01 of the learning rate, no more than the learning rate
            warmup_scaler = min(max((last_step - self.start) / self.warmup, 0.01), 1.0)
            lr_scale *= warmup_scaler
            
        return lr_scale

def build_scheduler(optimizer, cfg: DictConfig):
    
    cfg = cfg.copy()
    
    def get_scaler(cfg):
        return WarmupMultiStepScaler(
            start=cfg.start, 
            end=cfg.end, 
            milestones=[cfg.start + (cfg.end - cfg.start)*ms
                        for ms in cfg.lr_milestone], 
            gamma=cfg.lr_decrease, 
            warmup=cfg.warmup)
        
    config_groups = cfg.pop("groups", None)

    if config_groups is not None:
        # pop group-specific configs so the remainders are common configs
        group_specific_config = {g:OmegaConf.to_container(cfg.pop(g)) for g in config_groups}
        # override common config with group-specific configs
        for g in config_groups:
            group_specific_config[g] = OmegaConf.merge(cfg.copy(), group_specific_config[g])
        # create a scaler for each of the group
        scaler = [get_scaler(group_specific_config[g]) for g in config_groups]
    else:
        scaler = get_scaler(cfg)
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scaler)
    
    return scheduler