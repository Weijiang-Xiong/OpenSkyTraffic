import os 
import time
import shutil
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt

from omegaconf import DictConfig

import torch
import torch.optim as optim

from netsanut.engine.base import TrainerBase

from .base import HookBase, TrainerBase

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

class TrainingStageManager(HookBase):
    
    def __init__(self, milestone: int=None, config:Dict[int, DictConfig]=None, from_best=False) -> None:
        super().__init__()
        self.milestone = milestone
        self.config = config # the new config to use at the milestone epoch
        self.from_best = from_best # whether to load the model with best validation loss 
        
    def before_epoch(self, trainer: TrainerBase):
        
        if self.milestone is None:
            return 
        
        if trainer.epoch_num == self.milestone:
            trainer.model.adapt_to_new_config(self.config.model)
            if self.from_best:
                trainer.logger.info("Train uncertainty part based on previous best")
                ckpthook = [h for h in trainer._hooks if isinstance(h, CheckpointSaver)][0]
                trainer.load_checkpoint(ckpthook.best_ckpt_path, resume=False)
                
                
class ValidationHook(HookBase):
    # TODO add eval period option
    def __init__(self, eval_function:Callable) -> None:
        super().__init__()
        self.eval_function = eval_function
    
    def after_epoch(self, trainer: TrainerBase):
        ts = time.perf_counter()
        validation_metrics = self.eval_function(trainer.model)
        te = time.perf_counter()
        trainer.storage.put_scalar(name="epoch_inference_time", value=te-ts)
        trainer.storage.put_scalars(**validation_metrics, suffix="val")

class TestHook(HookBase):
    
    def __init__(self, eval_function:Callable) -> None:
        super().__init__()
        self.eval_function = eval_function

    def after_train(self, trainer: TrainerBase):
        ts = time.perf_counter()
        test_metrics = self.eval_function(trainer.model)
        te = time.perf_counter()
        trainer.storage.put_scalar(name="final_test_time", value=te-ts)
        trainer.storage.put_scalars(**test_metrics, suffix="test")

class StepBasedLRScheduler(HookBase):
    """ 
        adapted from detectron2.engine.hooks.LRScheduler
        Scheduler is an optional part in the training, some people use and some don't.
        So it's written as a hook, and the scheduler will be assigned to the trainer before training.
    """
    def __init__(self, scheduler) -> None:
        self.scheduler = scheduler
        
    def before_train(self, trainer: TrainerBase):
        trainer.scheduler = self.scheduler
        for scaler in trainer.scheduler.lr_lambdas:
            scaler.to_iteration_based(trainer.dataset_len)
            
    def after_step(self, trainer: TrainerBase):
        lrs = [group['lr'] for group in trainer.scheduler.optimizer.param_groups]
        trainer.scheduler.step()
        for idx, lr in enumerate(lrs):
            trainer.storage.put_scalar("lr_group_{}".format(idx), lr)
    
    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict['scheduler'])
    
                
class CheckpointSaver(HookBase):
    
    # TODO add period that corresponds to eval period
    # TODO add option for using a metric that is the higher the better
    def __init__(self, metric="mae_val", test_best_ckpt=True) -> None:
        super().__init__()
        self.metric = metric
        self.lowest_loss = np.inf
        self.best_ckpt_path = ""
        self.last_ckpt_path = ""
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
        self.last_ckpt_path = trainer.save_checkpoint(additional_note=note)
        
        if latest_loss < self.lowest_loss:
            self.lowest_loss = latest_loss
            self.best_ckpt_path = self.last_ckpt_path
        
    def after_train(self, trainer: TrainerBase):
        
        trainer.logger.info("The best achieved validation loss is {}".format(round(self.lowest_loss, 2)))
        
        # copy the best checkpoint, and name it
        if os.path.exists(self.best_ckpt_path):
            trainer.logger.info("The model with best validation loss is {}".format(self.best_ckpt_path))
            copy_path = "{}/model_bestval.pth".format(trainer.save_dir)
            shutil.copyfile(self.best_ckpt_path, copy_path)
            trainer.logger.info("Copying the best model to {}".format(copy_path))
            
        if self.test_best_ckpt:
            trainer.logger.info("Loading the model from {}".format(self.best_ckpt_path))
            trainer.load_checkpoint(self.best_ckpt_path, resume=False)
        
        copy_path = "{}/model_final.pth".format(trainer.save_dir)
        shutil.copyfile(self.last_ckpt_path, copy_path)
        trainer.logger.info("Copying the final model to {}".format(copy_path))

class MetricLogger(HookBase):
    
    # TODO study detectron2.utils.events.CommonMetricPrinter a improve this part
    def __init__(self) -> None:
        super().__init__()
        self.epoch_log = defaultdict(list)
    
    def after_epoch(self, trainer: TrainerBase):
        
        latest_log = trainer.storage.latest()
        trainer.logger.info('Epoch: {:03d}, Training Time: {:.4f} secs Inference Time: {:.4f} secs Train Loss {:.4f}'.format(
            trainer.epoch_num, 
            latest_log["epoch_train_time"][0], 
            latest_log["epoch_inference_time"][0],
            latest_log["loss"][0]
            )
        )
        trainer.logger.info("Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}".format(
            latest_log["mae_train"][0], latest_log["mape_train"][0], latest_log["rmse_train"][0]))
        trainer.logger.info("Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}".format(
            latest_log["mae_val"][0], latest_log["mape_val"][0], latest_log["rmse_val"][0]))

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
        for key, value in self.epoch_log.items():
            value.clear()

class GradientClipper(HookBase):
    
    def __init__(self, clip_value: float=None) -> None:
        super().__init__()
        self.clip_value: float = clip_value
        
        
    def after_backward(self, trainer: TrainerBase):
        
        if self.clip_value is not None:
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), self.clip_value)
            
            
class PlotTrainingLog(HookBase):
    
    def __init__(self, dpi=300, use_sns=True) -> None:
        super().__init__()
        self.dpi = dpi
        self.use_sns = use_sns

    def after_train(self, trainer: TrainerBase):
        if self.use_sns:
            import seaborn as sns 
            sns.set_style("darkgrid")
    
        stored_data = trainer.storage._history
        for key, value in stored_data.items():
            value_epoch = np.array(value.data())
            if len(value_epoch) == 1:
                continue
            value, epoch = value_epoch[:, 0], value_epoch[:, 1]
            
            fig, ax = plt.subplots() 
            ax.plot(epoch, value, label=key)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(key)
            ax.set_title("The progress of {} during training".format(key))
            ax.legend()
            
            save_name = "log_{}".format(key)
            fig.tight_layout()
            fig.savefig("{}/{}.pdf".format(trainer.save_dir, save_name), dpi=self.dpi)
            