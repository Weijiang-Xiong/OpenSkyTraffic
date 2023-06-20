import os 
import time
import shutil
from collections import Counter, defaultdict
from typing import List

import numpy as np
import torch
import torch.optim as optim

from netsanut.engine.base import TrainerBase

from .base import HookBase, TrainerBase


class UncertaintyTraining(HookBase):
    
    
    def __init__(self) -> None:
        super().__init__()
    
    def before_train(self, trainer: TrainerBase):
        return
    
    
class EpochTimer(HookBase):
    
    def __init__(self) -> None:
        super().__init__()
        self.te: float = 0.0
        self.ts: float = 0.0
    
    def before_epoch(self, trainer: TrainerBase):
        self.ts = time.perf_counter()
    
    def after_epoch(self, trainer: TrainerBase):
        self.te = time.perf_counter()
        trainer.storage.put_scalar(name="epoch_train_time", value=self.te-self.ts)
    
class ValidationHook(HookBase):
    # TODO add eval period option
    def __init__(self) -> None:
        super().__init__()
    
    def after_epoch(self, trainer: TrainerBase):
        ts = time.perf_counter()
        validation_metrics = trainer.evaluate(trainer.model, trainer.train_val_test_loaders['val'])
        te = time.perf_counter()
        trainer.storage.put_scalar(name="epoch_inference_time", value=te-ts)
        trainer.storage.put_scalars(**validation_metrics, suffix="val")

class TestHook(HookBase):
    
    def __init__(self) -> None:
        super().__init__()

    def after_train(self, trainer: TrainerBase):
        ts = time.perf_counter()
        test_metrics = trainer.evaluate(trainer.model, trainer.train_val_test_loaders['test'], verbose=True)
        te = time.perf_counter()
        trainer.storage.put_scalar(name="final_test_time", value=te-ts)
        trainer.storage.put_scalars(**test_metrics, suffix="test")

class LRScheduler(HookBase):
    """ 
        adapted from detectron2.engine.hooks.LRScheduler
        Scheduler is an optional part in the training, some people use and some don't.
        So it's written as a hook, and the scheduler will be assigned to the trainer before training.
    """
    def __init__(self, scheduler) -> None:
        self.scheduler = scheduler
        
    def before_train(self, trainer: TrainerBase):
        trainer.scheduler = self.scheduler
        self._best_param_group_id = LRScheduler.get_best_param_group_id(trainer.optimizer)
        
    def after_epoch(self, trainer: TrainerBase):
        # NOTE this could be problematic if different parameters have different learning rates 
        lr = trainer.scheduler.optimizer.param_groups[self._best_param_group_id]['lr']
        trainer.scheduler.step()
        trainer.storage.put_scalar("lr", lr)
    
    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict['scheduler'])
    
    @staticmethod
    def get_best_param_group_id(optimizer):
        
        # NOTE: some heuristics on what LR to summarize
        # summarize the param group with most parameters
        largest_group = max(len(g["params"]) for g in optimizer.param_groups)

        if largest_group == 1:
            # If all groups have one parameter,
            # then find the most common initial LR, and use it for summary
            lr_count = Counter([g["lr"] for g in optimizer.param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer.param_groups):
                if g["lr"] == lr:
                    return i
        else:
            for i, g in enumerate(optimizer.param_groups):
                if len(g["params"]) == largest_group:
                    return i
                
                
class CheckpointSaver(HookBase):
    
    # TODO add period that corresponds to eval period
    # TODO add option for using a metric that is the higher the better
    def __init__(self, metric="mae_val", test_best_ckpt=True) -> None:
        super().__init__()
        self.metric = metric
        self.lowest_loss = np.inf
        self.best_ckpt_path = ""
        self.test_best_ckpt = test_best_ckpt
        
    def after_epoch(self, trainer: TrainerBase):
        
        metric_tuple = trainer.storage.latest().get(self.metric)
        if metric_tuple is None:
            trainer.logger.warning(
                f"Given val metric {self.metric} does not seem to be computed/stored."
                "Will not be checkpointing based on it."
            )
            return
        else:
            latest_loss, latest_epoch = metric_tuple
        
        note = '{}_{}'.format(self.metric, round(latest_loss, 2))
        save_path = trainer.save_checkpoint(additional_note=note)
        
        if latest_loss < self.lowest_loss:
            self.lowest_loss = latest_loss
            self.best_ckpt_path = save_path
    
    def after_train(self, trainer: TrainerBase):
        
        trainer.logger.info("The valid loss on best model is {}".format(round(self.lowest_loss, 2)))
        
        # copy the best checkpoint, and name it
        if os.path.exists(self.best_ckpt_path):
            save_folder = os.path.dirname(self.best_ckpt_path)
            copy_path = "{}/model_best.pth".format(save_folder)
            shutil.copyfile(self.best_ckpt_path, copy_path)
            trainer.logger.info("Copying the model to {}".format(copy_path))
            
        if self.test_best_ckpt:
            trainer.load_checkpoint(self.best_ckpt_path, resume=False)
        

class MetricLogger(HookBase):
    
    # TODO study detectron2.utils.events.CommonMetricPrinter a improve this part
    def __init__(self) -> None:
        super().__init__()
        self.epoch_log = defaultdict(list)
    
    def after_epoch(self, trainer: TrainerBase):
        
        latest_log = trainer.storage.latest()
        trainer.logger.info('Epoch: {:03d}, Training Time: {:.4f} secs'.format(trainer.epoch_num, latest_log["epoch_train_time"][0]))
        trainer.logger.info('Epoch: {:03d}, Inference Time: {:.4f} secs'.format(trainer.epoch_num, latest_log["epoch_inference_time"][0]))
        trainer.logger.info("Epoch: {:03d}, Train Loss {:.4f}".format(trainer.epoch_num, latest_log["loss"][0]))
        trainer.logger.info("Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}".format(
            latest_log["mae_train"][0], latest_log["mape_train"][0], latest_log["rmse_train"][0]))
        trainer.logger.info("Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}".format(
            latest_log["mae_val"][0], latest_log["mae_val"][0], latest_log["mae_val"][0]))

    def after_train(self, trainer: TrainerBase):

        trainer.logger.info("Training finished")
        trainer.logger.info("Average Training Time: {:.4f} secs/epoch".format(np.mean(trainer.storage["epoch_train_time"].values())))
        trainer.logger.info("Average Inference Time: {:.4f} secs/epoch".format(np.mean(trainer.storage["epoch_inference_time"].values())))


class TrainMetricRecorder(HookBase):
    
    def __init__(self) -> None:
        super().__init__()
        self.epoch_log = defaultdict(list)
        
    def after_step(self, trainer: TrainerBase):
        try:
            aux_metrics = trainer.model.pop_auxiliary_metrics()
        except:
            aux_metrics = dict()
        for k, v in aux_metrics.items():
            self.epoch_log[k].append(v)

    def after_epoch(self, trainer: TrainerBase):
        train_epoch_metrics = {key: np.mean(value) for key, value in self.epoch_log.items()}
        trainer.storage.put_scalars(suffix="train", **train_epoch_metrics)

class GradientClipper(HookBase):
    
    def __init__(self, clip_value: float=None) -> None:
        super().__init__()
        self.clip_value: float = clip_value
        
        
    def after_backward(self, trainer: TrainerBase):
        
        if self.clip_value is not None:
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), self.clip_value)