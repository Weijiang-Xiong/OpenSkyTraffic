import os
import time
import glob
import logging 

import torch
import torch.nn as nn 
import torch.optim as optim

import util
from model import *
from textwrap import dedent
from events import EventStorage, get_event_storage
from collections import defaultdict

class DefaultTrainer():
    """ The trainer takes care of the general logic in the training progress, such as 
            1. train on trainset, validate on validation set, and log the best performance
               on validation set, evaluate it on test set
            2. save and load checkpoint, resume from previous training
            3. running optimizer and scheduler 
    """
    
    def __init__(self, args, model:NTSModel, dataloader, datascaler):
        
        self.logger = logging.getLogger("default")
        
        self.model = model
        
        self.datascaler = datascaler
        self.dataloader = dataloader
        
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=args.learning_rate,
                                    weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, 
                                                        [int(args.max_epoch*ms) for ms in args.lr_milestone], 
                                                        gamma=args.lr_decrease)
        self.clip = args.grad_clip

        self.start_epoch = 0 # start epoch (may not be zero if resume from previous training)
        self.epoch_num = self.start_epoch
        self.max_epoch = args.max_epoch # max training epoch
        self.storage:EventStorage
        
        self.save_dir = args.save_dir if getattr(args, 'save_dir', None) else "./checkpoint"
        
        if args.resume or args.load_weights:
            self.load_checkpoint(args.checkpoint, resume=args.resume)


    def train(self):
        
        self.logger.info("start training...")
        self.logger.info("The model structure \n {}".format(self.model))
        
        with EventStorage() as self.storage:
            for epoch_num in range(self.start_epoch, self.max_epoch):
                
                self.epoch_num = epoch_num
                self.storage.iteration = epoch_num
                
                # training
                ts = time.perf_counter()
                train_metrics = self.train_epoch()
                te = time.perf_counter()
                self.logger.info('Epoch: {:03d}, Training Time: {:.4f} secs'.format(epoch_num, (te-ts)))
                self.storage.put_scalar(name="epoch_train_time", value=te-ts)
                self.storage.put_scalars(**train_metrics, suffix="train")
                
                # validation
                ts = time.perf_counter()
                validation_metrics = self.evaluate(self.model, self.dataloader['val_loader'])
                te = time.perf_counter()
                self.logger.info('Epoch: {:03d}, Inference Time: {:.4f} secs'.format(epoch_num, (te-ts)))
                self.storage.put_scalar(name="epoch_inference_time", value=te-ts)
                self.storage.put_scalars(**validation_metrics, suffix="val")
                
                self.scheduler.step()
                self.save_checkpoint(additional_note=round(validation_metrics["mae"], 2))
                
                self.logger.info("Epoch: {:03d}, Train Loss {:.4f}".format(epoch_num, train_metrics['loss']))
                self.logger.info("Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}".format(
                    train_metrics['mae'], train_metrics['mape'], train_metrics['rmse']))
                self.logger.info("Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}".format(
                    validation_metrics['mae'], validation_metrics['mape'], validation_metrics['rmse']))
                self.logger.info("Training Time: {:.4f}/epoch".format(self.storage.latest()['epoch_train_time'][0]))
            
            validation_loss_log = self.storage["mae_val"].values() # no need for iteration
            bestid = np.argmin(validation_loss_log)

            
            self.logger.info("Training finished")
            self.logger.info("The valid loss on best model is {}".format(round(validation_loss_log[bestid], 2)))
            self.logger.info("Average Training Time: {:.4f} secs/epoch".format(np.mean(self.storage["epoch_train_time"].values())))
            self.logger.info("Average Inference Time: {:.4f} secs/epoch".format(np.mean(self.storage["epoch_inference_time"].values())))
            
            # evaluate the best model on the test set
            best_model_path = self.format_file_path(list(range(self.start_epoch, self.max_epoch))[bestid], 
                                                    additional_note=round(validation_loss_log[bestid], 2))
            self.load_checkpoint(best_model_path, resume=False)
            
            return self.evaluate(self.model, 
                                self.dataloader['test_loader'], 
                                verbose=True)

    def train_epoch(self):
        
        self.model.train()
        
        epoch_log = defaultdict(list)
        
        self.dataloader['train_loader'].shuffle()
        for data, label in self.dataloader['train_loader'].get_iterator():
            
            self.optimizer.zero_grad()
            
            loss_dict = self.model(data, label) # out shape (N, M, T)
            loss = sum(loss_dict.values())
            loss.backward() 
            
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            
            aux_metrics = self.model.pop_auxiliary_metrics() if getattr(self.model, "record_auxiliary_metrics", False) else dict()
            aux_metrics.update({k:v.item() for k, v in loss_dict.items()})
            for key, value in aux_metrics.items():
                epoch_log[key].append(value)
        
        overall_metrics = {key:np.mean(value) for key, value in epoch_log.items()}
        
        return overall_metrics

    @staticmethod
    def evaluate(model:NTSModel, dataloader, verbose=False):
        
        logger = logging.getLogger("default")

        model.eval()
        
        all_preds, all_labels = [], []
        for iteration, (data, label) in enumerate(dataloader.get_iterator()):
            preds = model(data)
            
            all_preds.append(preds)
            all_labels.append(label)

        all_preds = torch.cat(all_preds, dim=0).cpu()
        all_labels = torch.cat(all_labels, dim=0).cpu()
        
        if verbose:
            logger.info("The shape of predicted {} and label {}".format(all_preds.shape, all_labels.shape))
            
        for i in range(12): # number of predicted time step 
            pred = all_preds[..., i]
            real = all_labels[..., i]
            aux_metrics = util.default_metrics(pred, real)

            if verbose:
                logger.info('Evaluate model on test data at {:d} time step'.format(i+1))
                logger.info('Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'.format(
                    aux_metrics['mae'], aux_metrics['mape'], aux_metrics['rmse']
                    )
                )

        overall_metrics = util.default_metrics(all_preds, all_labels)
        
        if verbose:
            logger.info('On average over 12 different time steps')
            logger.info('Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'.format(
                overall_metrics['mae'], overall_metrics['mape'], overall_metrics['rmse']
                )
            )
        
        return overall_metrics

    def load_checkpoint(self, ckpt_path:str, resume=False):
        """ load a checkpoint that contains model, optimizer, scheduler and epoch number 

        Args:
            ckpt_path (str): the path to the checkpoint
            resume (bool, optional): If set to true, then load everything from the checkpoint, and resume from previous training. Defaults to False, and only load the model weights
        """
        if not os.path.exists(ckpt_path):
            self.logger.warning("File not exists, skip loading {}!".format(ckpt_path))
            return 
        
        state_dict:dict = torch.load(ckpt_path)
        
        if resume: # load everything
            self.logger.info("Resuming from checkpoint {}".format(ckpt_path))
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.scheduler.load_state_dict(state_dict['scheduler'])
            # the model is saved after `epoch_num`, so start from the next one 
            self.start_epoch = state_dict['epoch_num'] + 1 
            self.epoch_num = self.start_epoch
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

    def save_checkpoint(self, additional_note:str=None):
        
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
        
        save_file_name = self.format_file_path(self.epoch_num, additional_note)
            
        torch.save(state_dict, save_file_name)
        self.logger.info("Checkpoint saved to {}".format(save_file_name))