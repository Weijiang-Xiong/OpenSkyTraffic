import os
import copy 
import time
import logging

from typing import Dict, List, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .base import TrainerBase

class DefaultTrainer(TrainerBase):
    """ Implement checkpoint saving and loading
    """

    def __init__(self, model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, max_epoch: int = 30, output_dir: str = None):
        
        super().__init__() 
        
        self.logger = logging.getLogger("default")

        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        
        self.start_epoch = 0  # start epoch (may not be zero if resume from previous training)
        self.max_epoch = max_epoch  # max training epoch
        
        self.save_dir = output_dir if output_dir is not None else "./checkpoint"

    @staticmethod
    def load_file(ckpt_path:str) -> dict:
        logger = logging.getLogger("default")
        if not os.path.exists(ckpt_path):
            logger.info("Checkpoint does not exist, train from scratch instead. Given Path: {}".format(ckpt_path if ckpt_path else None))
            return

        state_dict: dict = torch.load(ckpt_path)
        return state_dict
        
    def load_checkpoint(self, ckpt_path: str, resume=False):
        """ load a checkpoint that contains model, optimizer, scheduler and epoch number 

        Args:
            ckpt_path (str): the path to the checkpoint
            resume (bool, optional): If set to true, then load everything from the checkpoint, and resume from previous training. Defaults to False, and only load the model weights
        """
        state_dict = self.load_file(ckpt_path)
        
        if state_dict is None:
            return 

        if resume:  # load everything
            self.logger.info("Resuming from checkpoint {}".format(ckpt_path))
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
            # the model is saved after `epoch_num`, so start from the next one
            self.start_epoch = state_dict['epoch_num'] + 1
            for h in self._hooks:
                h.load_state_dict(state_dict)

        else:
            self.logger.info("Loading model weights from {}".format(ckpt_path))
            incompatible = self.model.load_state_dict(state_dict['model'], strict=False)
            if incompatible is not None:
                self.logger.info("Incompatible keys are : \n {}".format(incompatible))

        self.logger.info("Successfully loaded checkpoint file {}!".format(ckpt_path))

    def format_file_path(self, epoch_num, additional_note=None):

        save_file_name = "{}/{}_epoch_{}".format(self.save_dir, self.model.__class__.__name__.lower(), epoch_num)

        if additional_note is not None:
            save_file_name += "_{}".format(additional_note)

        save_file_name += ".pth"

        return save_file_name

    def save_checkpoint(self, additional_note: str = None):

        if not isinstance(self.model, nn.Module):
            self.logger.warning("The model is not properly initialized, will save None to state dict")
        if not isinstance(self.optimizer, optim.Optimizer):
            self.logger.warning("The optimizer is not properly initialized, will save None to state dict")
        if not isinstance(self.scheduler, optim.lr_scheduler.LRScheduler):
            self.logger.warning("The scheduler is not properly initialized, will save None to state dict")

        state_dict = {
            "model": self.model.state_dict() if isinstance(self.model, nn.Module) else None,
            "optimizer": self.optimizer.state_dict() if isinstance(self.optimizer, optim.Optimizer) else None,
            "scheduler": self.scheduler.state_dict() if isinstance(self.scheduler, optim.lr_scheduler.LRScheduler) else None,
            "epoch_num": self.epoch_num,
        }

        save_path = self.format_file_path(self.epoch_num, additional_note)

        torch.save(state_dict, save_path)
        self.logger.info("Epoch: {:03d}, Checkpoint saved to {}".format(self.epoch_num, save_path))
        return save_path