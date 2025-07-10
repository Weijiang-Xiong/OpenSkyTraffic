import torch.optim as optim

from typing import Any
from collections import Counter
from bisect import bisect_right

from omegaconf import OmegaConf, DictConfig
class WarmupMultiStepScaler:
    
    """ A scaler function for learning rate scheduler.
        The learning rate gradually increase from 1% to 100% of the chosen value in the first a
        few training epochs specified by `warmup`. The learning rate will decrease to `gamma` of previous value when the training reaches a milestone. 
        The learning rate is set to zero if the training progress is not in [start, end], which
        allows one to set different learning rate schedules for different parts of a model. 
    """
    
    def __init__(self, start:int, end:int, milestones:list, gamma:float=0.1, warmup:float=None) -> None:
        
        self.start = start
        self.end = end
        _milestones = [start + (end - start) * m for m in milestones]
        self.milestones = Counter(_milestones)
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

