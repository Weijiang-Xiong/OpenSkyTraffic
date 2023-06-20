import os
import time
import logging

from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from netsanut.engine.base import HookBase

from netsanut.util import default_metrics
from netsanut.events import EventStorage
from netsanut.data import TensorDataScaler

from . import hooks
from .base import HookBase, TrainerBase

class DefaultTrainer(TrainerBase):
    """ The trainer takes care of the general logic in the training progress, such as 
            1. train on trainset, validate on validation set, and log the best performance
               on validation set, evaluate it on test set
            2. save and load checkpoint, resume from previous training
            3. running optimizer and scheduler 
    """

    def __init__(self, cfg, model: nn.Module, dataloaders: Dict[str, DataLoader]):
        
        super().__init__() 
        
        self.logger = logging.getLogger("default")

        self.model = model

        self.train_val_test_loaders = dataloaders

        self.optimizer = self.build_optimizer(self.model.parameters(), cfg)

        self.register_hooks(self.build_hooks(cfg))
        
        self.start_epoch = 0  # start epoch (may not be zero if resume from previous training)
        self.max_epoch = cfg.train.max_epoch  # max training epoch
        
        self.save_dir = cfg.train.output_dir if getattr(cfg.train, 'output_dir', None) else "./checkpoint"

    @staticmethod
    def evaluate(model: nn.Module, dataloader: DataLoader, verbose=False):

        logger = logging.getLogger("default")

        model.eval()

        all_preds, all_labels = [], []
        for data, label in dataloader:

            data, label = data.cuda(), label.cuda()

            preds = model(data)

            all_preds.append(preds)
            all_labels.append(label)

        all_preds = torch.cat(all_preds, dim=0).cpu()
        all_labels = torch.cat(all_labels, dim=0).cpu()

        if verbose:
            logger.info("The shape of predicted {} and label {}".format(all_preds.shape, all_labels.shape))

        for i in range(12):  # number of predicted time step
            pred = all_preds[:, i, :]
            real = all_labels[:, i, :]
            aux_metrics = default_metrics(pred, real)

            if verbose:
                logger.info('Evaluate model on test data at {:d} time step'.format(i+1))
                logger.info('Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'.format(
                    aux_metrics['mae'], aux_metrics['mape'], aux_metrics['rmse']
                )
                )

        overall_metrics = default_metrics(all_preds, all_labels)

        if verbose:
            logger.info('On average over 12 different time steps')
            logger.info('Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'.format(
                overall_metrics['mae'], overall_metrics['mape'], overall_metrics['rmse']
            )
            )

        return overall_metrics

    def load_checkpoint(self, ckpt_path: str, resume=False):
        """ load a checkpoint that contains model, optimizer, scheduler and epoch number 

        Args:
            ckpt_path (str): the path to the checkpoint
            resume (bool, optional): If set to true, then load everything from the checkpoint, and resume from previous training. Defaults to False, and only load the model weights
        """
        if not os.path.exists(ckpt_path):
            self.logger.info("Checkpoint does not exist, train from scratch instead. Given Path: {}".format(ckpt_path if ckpt_path else None))
            return

        state_dict: dict = torch.load(ckpt_path)

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
    
    @staticmethod
    def build_scheduler(optimizer, cfg):
        
        milestones=[int(cfg.train.max_epoch*ms) for ms in cfg.scheduler.lr_milestone]
        gamma=cfg.scheduler.lr_decrease
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma
        )
        return scheduler
    
    @staticmethod
    def build_optimizer(params, cfg):
        optimizer = optim.Adam(
            params,
            lr=cfg.optimizer.learning_rate,
            weight_decay=cfg.optimizer.weight_decay
        )
        return optimizer
    
    def build_hooks(self, cfg) -> List[HookBase]:
        ret = [
            hooks.EpochTimer(),
            hooks.TrainMetricRecorder(),
            hooks.LRScheduler(scheduler=self.build_scheduler(self.optimizer, cfg)),
            hooks.ValidationHook(),
            hooks.CheckpointSaver(test_best_ckpt=cfg.train.test_best_ckpt),
            hooks.MetricLogger(),
            hooks.TestHook(),
            hooks.GradientClipper(clip_value=cfg.optimizer.grad_clip),
        ]
        
        return ret