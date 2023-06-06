""" modified from https://github.com/nnzhan/Graph-WaveNet/blob/master/util.py
"""

import torch
import logging 

import numpy as np

def masked_mse(preds, labels, null_val=np.nan):
    report_nan(preds)
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    report_nan(preds)
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    report_nan(preds)
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)



def masked_mape(preds, labels, null_val=np.nan):
    report_nan(preds)
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def default_metrics(pred, label):
    
    mae = masked_mae(pred,label,0.0).item()
    mape = masked_mape(pred,label,0.0).item()
    rmse = masked_rmse(pred,label,0.0).item()
    
    return {"mae":mae, "mape":mape, "rmse":rmse}


def report_nan(res:torch.tensor, use_logger=True):
    logger = logging.getLogger("default")
    num_of_nan = torch.sum(torch.isnan(res)).item()
    if num_of_nan > 0:
        if not use_logger:
            print("number of nan in result: {}".format(torch.sum(torch.isnan(res)).item()))
        else:
            logger.warning("number of nan in result: {}".format(torch.sum(torch.isnan(res)).item()))