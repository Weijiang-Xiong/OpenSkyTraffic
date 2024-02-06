import os
import time
import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from netsanut.events import EventStorage


class HookBase:
    """ This class is put here not hooks.py because the type hints creates a circular dependency
        a walk-around is use string 
        things will work like:
    
        hook.before_train() 
        for epoch in range(max_epoch):
            hook.before_epoch()
            for data in dataloader: 
                hook.before_step()
                trainer.run_step() 
                hook.after_step() 
            hook.after_epoch()
        hook.after_train()
        
        Note: 
            1. the loop over dataloader will be defined in trainer.train_epoch()
            2. hook.after_backward() will be called within the run_step
            3. resume will be called in trainer.load_checkpoint()
    """

    def __init__(self) -> None:
        pass

    def before_train(self, trainer: "TrainerBase"):
        pass

    def after_train(self, trainer: "TrainerBase"):
        pass

    def before_epoch(self, trainer: "TrainerBase"):
        pass

    def after_epoch(self, trainer: "TrainerBase"):
        pass

    def before_step(self, trainer: "TrainerBase"):
        pass

    def after_step(self, trainer: "TrainerBase"):
        pass

    def after_backward(self, trainer: "TrainerBase"):
        pass

    def load_state_dict(self, state_dict):
        pass


class TrainerBase:

    """ A base trainer class that:
            1. runs a training loop
            2. calls hooks
        
        It also has interface for checkpoint loading and saving, but not implementation
    """

    def __init__(self) -> None:

        self.logger: logging.Logger

        self.model: nn.Module
        self.dataloader: DataLoader
        self.optimizer: optim.Optimizer
        self.scheduler: Optional[optim.lr_scheduler.LambdaLR]

        self._hooks: List[HookBase] = []
        self.epoch_num: int = 0
        self.start_epoch: int = 0
        self.max_epoch: int
        self.storage: EventStorage
        self.save_dir: str
        
        # these are related to logging within an epoch
        self.batch_idx: int  # the index of current training batch (to compute progress)

    @property
    def dataset_len(self):
        return len(self.dataloader)
    
    @property
    def epoch_progress(self):
        return (self.batch_idx + 1) / self.dataset_len
    
    def train(self):

        self.logger.info("start training...")
        self.logger.info("The model structure \n {}".format(self.model))

        with EventStorage() as self.storage:
            try:
                self.before_train()
                for self.epoch_num in range(self.start_epoch, self.max_epoch):
                    self.before_epoch()
                    self.train_epoch()
                    self.after_epoch()
            except Exception:
                self.logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def train_epoch(self):

        self.model.train()

        self.loss_log = defaultdict(list)

        for self.batch_idx, data in enumerate(self.dataloader):

            self.before_step()

            loss_dict = self.model(data)
            loss = sum(loss_dict.values())
            loss.backward()
            self.after_backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            for k, v in loss_dict.items():
                self.loss_log[k].append(v.item())

            self.after_step()

        self.storage.put_scalars(**{key: np.mean(value) for key, value in self.loss_log.items()})

    def load_checkpoint(self, ckpt_path: str, resume=False):
        raise NotImplementedError

    def save_checkpoint(self, additional_note: str = None) -> str:
        raise NotImplementedError

    def register_hooks(self, hooks: List[HookBase]):
        """ The hooks will be called in the same order as they are registered
            It's possible to add a priority attribute for each method, and sort the hooks according to priority.
            But this will introduce too much overhead.
        """
        hooks = [h for h in hooks if h is not None]
        self._hooks.extend(hooks)

    def build_hooks(self, cfg) -> List[HookBase]:

        raise NotImplementedError

    def before_train(self):
        for hook in self._hooks:
            hook.before_train(self)

    def after_train(self):
        self.storage.epoch_num = self.epoch_num
        if self.epoch_num + self.epoch_progress != self.max_epoch:
            self.logger.info("Training did not reach max epoch, check the log if this is not expected.")
        for hook in self._hooks:
            hook.after_train(self)

    def before_epoch(self):
        self.storage.epoch_num = self.epoch_num
        for hook in self._hooks:
            hook.before_epoch(self)

    def after_epoch(self):
        for hook in self._hooks:
            hook.after_epoch(self)

    def before_step(self):
        self.storage.epoch_progress = self.epoch_progress
        for hook in self._hooks:
            hook.before_step(self)

    def after_step(self):
        self.storage.epoch_progress = self.epoch_progress
        for hook in self._hooks:
            hook.after_step(self)

    def after_backward(self):
        for hook in self._hooks:
            hook.after_backward(self)
