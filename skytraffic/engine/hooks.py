import os 
import json
import time
import shutil
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt

from omegaconf import DictConfig

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

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


class EvalHook(HookBase):
    # TODO add eval period option
    def __init__(self, 
                 eval_function:Callable, 
                 metric_suffix="eval", 
                 period: int = 0,
                 eval_after_train: bool =True) -> None:
        super().__init__()
        self.eval_function = eval_function
        self.suffix = metric_suffix
        self.period = period # no per-epoch evaluation by default
        self.eval_after_train = eval_after_train
    
    def after_epoch(self, trainer: TrainerBase):
        if self.period <= 0:
            return
        elif self.period > 1 and (trainer.epoch_num + 1) % self.period != 0:
            return
        # if eval_per_epoch is 1, we evaluate every epoch
        
        ts = time.perf_counter()
        res = self.eval_function(trainer.model)
        te = time.perf_counter()
        trainer.storage.put_scalar(name="epoch_inference_time", value=te-ts)
        if isinstance(res, dict):
            trainer.storage.put_scalars(**res, suffix=self.suffix)
        trainer.logger.info("Evaluation Metrics ({}): {}".format(
            self.suffix,
            "  ".join(["{}: {:.4f}".format(k, v) for k, v in res.items()]),
        ))
        
    def after_train(self, trainer: TrainerBase):
        if not self.eval_after_train:
            return
        
        ts = time.perf_counter()
        test_metrics = self.eval_function(trainer.model)
        te = time.perf_counter()
        trainer.storage.put_scalar(name="final_test_time", value=te-ts)
        if isinstance(test_metrics, dict):
            trainer.storage.put_scalars(**test_metrics, suffix=self.suffix)
        

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
    def __init__(self, metric:str=None, test_best_ckpt:bool=True, period=5, verbose=False) -> None:
        super().__init__()
        self.metric:str = metric
        self.lowest_loss:float = np.inf
        self.best_ckpt_path: str = ""
        self.last_ckpt_path: str = ""
        self.test_best_ckpt = test_best_ckpt
        self.period = period
        self.verbose = verbose
        
    def after_epoch(self, trainer: TrainerBase):
        if not (trainer.epoch_num % self.period == 0 and trainer.epoch_num > 0):
            return
        
        metric_this_epoch = trainer.storage.latest().get(self.metric) if self.metric is not None else None
        
        # select a best checkpoint based on the metricif it is available,
        # otherwise just save the latest checkpoint
        if metric_this_epoch is None:
            if self.verbose:
                trainer.logger.info(
                    "Selection metric {} is not available, skip checkpoint selection. available metrics are {}".format(self.metric, trainer.storage.latest().keys())
                )
            self.test_best_ckpt = False
            self.last_ckpt_path = trainer.save_checkpoint()
        else:
            latest_loss, latest_epoch = metric_this_epoch
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
        
        # load the best checkpoint for testing (in EvalHook)
        if self.test_best_ckpt:
            trainer.logger.info("Loading the model from {}".format(self.best_ckpt_path))
            trainer.load_checkpoint(self.best_ckpt_path, resume=False)
        
        final_model_path = trainer.save_checkpoint()
        copy_path = "{}/model_final.pth".format(trainer.save_dir)
        shutil.copyfile(final_model_path, copy_path)
        trainer.logger.info("Copying the final model to {}".format(copy_path))

class MetricPrinter(HookBase):
    
    # TODO study detectron2.utils.events.CommonMetricPrinter a improve this part
    def __init__(self) -> None:
        super().__init__()
        self.epoch_log = defaultdict(list)
    
    def after_epoch(self, trainer: TrainerBase):
        
        latest_log = trainer.storage.latest()
        trainer.logger.info('Epoch: {:03d}, Training Time: {:.4f} secs Inference Time: {:.4f} secs'.format(
            trainer.epoch_num, 
            latest_log["epoch_train_time"][0], 
            latest_log["epoch_inference_time"][0],
            )
        )
        trainer.logger.info("Training Losses: " +"  ".join([
            "{} {:.4f}".format(k, v[0]) 
            for k, v in latest_log.items()
            if "loss" in k
        ]))

    def after_train(self, trainer: TrainerBase):

        trainer.logger.info("Training finished")
        trainer.logger.info("Average Training Time: {:.4f} secs/epoch".format(np.mean(trainer.storage["epoch_train_time"].values())))
        trainer.logger.info("Average Inference Time: {:.4f} secs/epoch".format(np.mean(trainer.storage["epoch_inference_time"].values())))


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
        self.figure_dir = ""

    def before_train(self, trainer: TrainerBase):
        # create a directory to save the figures
        self.figure_dir = "{}/figures".format(trainer.save_dir)
        if not os.path.exists(self.figure_dir):
            os.makedirs(self.figure_dir, exist_ok=False)
        
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
            fig.savefig("{}/{}.pdf".format(self.figure_dir, save_name), dpi=self.dpi)
            
            # also save the data as a json file
            with open("{}/{}.json".format(self.figure_dir, save_name), "w") as f:
                json.dump({"epoch": epoch.tolist(), "value": value.tolist()}, f)
            

class TensorboardWriter(HookBase):
    """
        Write the training log to tensorboard, the period should be the same as the evaluation period
        This writer write the metrics every a few epochs, so the step-wise metrics will not be completely recorded.
    """
    def __init__(self, save_dir: str, period: int = 0) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.period = period
        self.writer = SummaryWriter(log_dir=self.save_dir)
        
    def after_epoch(self, trainer: TrainerBase):
        if self.period <= 0:
            return
        elif self.period > 1 and (trainer.epoch_num + 1) % self.period != 0:
            return
        
        for key, (value, epoch_num) in trainer.storage.latest().items():
            self.writer.add_scalar(key, value, epoch_num)

    def after_train(self, trainer: TrainerBase):
        self.writer.close()
