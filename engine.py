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

class DefaultTrainer():
    
    def __init__(self, args, model, dataloader, datascaler):
        
        self.logger = logging.getLogger("default")
        
        self.model = model
        self.loss = util.masked_mae
        
        self.scaler = datascaler
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
        self.device = torch.device(args.device)

        self.save_dir = args.save_dir if getattr(args, 'save_dir', None) else "./checkpoint"
        
        if args.load_weights:
            self.load_checkpoint(args.checkpoint, args.resume)
            
        
        
    def train_step(self, data, label):
        
        self.model.train()
        self.optimizer.zero_grad()
        
        output, _ = self.model(data) # out shape (N, M, T)
        predict = self.scaler.inverse_transform(output)
        
        loss = self.loss(predict, label, 0.0)
        loss.backward()
        
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        
        mape = util.masked_mape(predict,label,0.0).item()
        mae = util.masked_mae(predict,label,0.0).item()
        rmse = util.masked_rmse(predict,label,0.0).item()
        
        return mae,mape,rmse

    def eval_step(self, data, label):
        
        self.model.eval()
        
        output, _ = self.model(data)
        predict = self.scaler.inverse_transform(output)
        
        mape = util.masked_mape(predict,label,0.0).item()
        mae = util.masked_mae(predict,label,0.0).item()
        rmse = util.masked_rmse(predict,label,0.0).item()
        
        return mae,mape,rmse

    def train(self):
        
        self.logger.info("start training...")
        self.logger.info("The model structure \n {}".format(self.model))
        
        validation_loss_log = []
        val_time = []
        train_time = []
        for epoch_num in range(self.start_epoch, self.max_epoch):
            
            self.epoch_num = epoch_num
            
            train_loss, train_mape, train_rmse = [[] for _ in range(3)]
            t1 = time.time()
            self.dataloader['train_loader'].shuffle()
            
            for iteration, (x, y) in enumerate(self.dataloader['train_loader'].get_iterator(), start=1):
                trainx = torch.Tensor(x).to(self.device)
                trainx = trainx.transpose(1, 3)
                trainy = torch.Tensor(y).to(self.device)
                trainy = trainy.transpose(1, 3)
                metrics = self.train_step(trainx, trainy[:, 0, :, :])
                train_loss.append(metrics[0])
                train_mape.append(metrics[1])
                train_rmse.append(metrics[2])
                    
            t2 = time.time()
            train_time.append(t2-t1)

            # validation
            valid_loss = []
            valid_mape = []
            valid_rmse = []
            s1 = time.time()
            for iteration, (x, y) in enumerate(self.dataloader['val_loader'].get_iterator()):
                testx = torch.Tensor(x).to(self.device)
                testx = testx.transpose(1, 3)
                testy = torch.Tensor(y).to(self.device)
                testy = testy.transpose(1, 3)
                metrics = self.eval_step(testx, testy[:, 0, :, :])
                valid_loss.append(metrics[0])
                valid_mape.append(metrics[1])
                valid_rmse.append(metrics[2])
            s2 = time.time()
            self.logger.info('Epoch: {:03d}, Inference Time: {:.4f} secs'.format(epoch_num, (s2-s1)))
            val_time.append(s2-s1)
            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            validation_loss_log.append(mvalid_loss)

            self.logger.info("Epoch: {:03d}".format(epoch_num))
            self.logger.info("Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}".format(
                mtrain_loss, mtrain_mape, mtrain_rmse))
            self.logger.info("Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}".format(
                mvalid_loss, mvalid_mape, mvalid_rmse))
            self.logger.info("Training Time: {:.4f}/epoch".format((t2-t1)))

            self.scheduler.step()
            self.save_checkpoint(additional_note=round(mvalid_loss.item(), 2))
            
        bestid = np.argmin(validation_loss_log)
        
        self.logger.info("Training finished")
        self.logger.info("The valid loss on best model is {}".format(round(validation_loss_log[bestid], 2)))
        self.logger.info("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        self.logger.info("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
        
        # evaluate the best model on the test set
        best_model_path = self.format_file_path(list(range(self.start_epoch, self.max_epoch))[bestid], 
                                                additional_note=round(validation_loss_log[bestid], 2))
        self.load_checkpoint(best_model_path, resume=False)
        
        return self.evaluate(self.model, 
                             self.dataloader['test_loader'], 
                             self.scaler)
    
    def evaluate(self, model, dataloader, scaler):
        
        outputs = []
        realy = []
        for iter, (x, y) in enumerate(dataloader.get_iterator()):
            testx = torch.Tensor(x).to(self.device).transpose(1, 3)
            testy = torch.Tensor(y).to(self.device).transpose(1, 3)[:, 0, :, :]
            with torch.no_grad():
                preds, var = model(testx)
            outputs.append(preds)
            realy.append(testy)

        yhat = torch.cat(outputs, dim=0)
        yhat = scaler.inverse_transform(yhat)
        realy = torch.cat(realy, dim=0)

        amae = []
        amape = []
        armse = []
        self.logger.info("The shape of predicted {} and label {}".format(yhat.shape, realy.shape))
        for i in range(12): # number of predicted time step 
            pred = yhat[..., i]
            real = realy[..., i]
            metrics = util.metric(pred, real)
            self.logger.info('Evaluate best model on test data at {:d} time step, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'.format(i+1, metrics[0], metrics[1], metrics[2]))
            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])

        self.logger.info(
            'On average over 12 different time steps, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'.format(*util.metric(yhat, realy))
        )

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
            self.model.load_state_dict(state_dict['model'])
            
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