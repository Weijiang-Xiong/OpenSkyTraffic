import torch.optim as optim
from model import *
import util
import torch
import torch.nn as nn 

class Trainer():
    def __init__(self, scaler, supports, args):
        
        self.model = TTNet(dropout=args.dropout, 
                           supports=supports, 
                           in_dim=args.in_dim, 
                           out_dim=args.pred_win, 
                           hid_dim=args.hid_dim, 
                           enc_layers=args.enc, 
                           dec_layers=args.dec, 
                           heads=args.nhead)
        
        self.model.to(args.device)
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=args.learning_rate,
                                    weight_decay=args.weight_decay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 3

    def train(self, input, real_val):
        
        self.model.train()
        self.optimizer.zero_grad()
        
        output = self.model(input) # out shape (N, T, M, 1)
        output = output.transpose(1, 3) # out shape (N, 1, M, T)
        
        real = torch.unsqueeze(real_val,dim=1) 
        predict = self.scaler.inverse_transform(output)
        
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        
        mape = util.masked_mape(predict,real,0.0).item()
        mae = util.masked_mae(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        
        return mae,mape,rmse

    def eval(self, input, real_val):
        
        self.model.eval()
        
        output = self.model(input)
        output = output.transpose(1, 3)
        
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        
        mape = util.masked_mape(predict,real,0.0).item()
        mae = util.masked_mae(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        
        return mae,mape,rmse
