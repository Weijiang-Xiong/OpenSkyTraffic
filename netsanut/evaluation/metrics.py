""" masked MAE, MAPE, RMSE are modified from https://github.com/nnzhan/Graph-WaveNet/blob/master/util.py
"""

import torch
import logging 
from typing import Dict

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

def uncertainty_metrics(pred: torch.Tensor, target: torch.Tensor, scale:torch.Tensor, 
                        offset_coeffs: Dict[float, float], ignore_value=0.0, verbose=False):
    pred, target, scale = pred.flatten(), target.flatten(), scale.flatten()
    if ignore_value is not None:
        valid = (target != ignore_value)
        pred, target, scale = pred[valid], target[valid], scale[valid]
    
    Conf, OC_val, CP, CCE, AO = [[] for _ in range(5)]
    for confidence, offset in offset_coeffs.items():
        ub = pred + offset * scale
        lb = pred - offset * scale
        
        covered = torch.logical_and(target < ub, target > lb)
        coverage_percentage = covered.sum() / covered.numel()
        
        Conf.append(confidence)
        OC_val.append(offset)
        CP.append(coverage_percentage.item())
        CCE.append(abs((confidence - coverage_percentage).item()))
        AO.append(torch.mean(offset*scale).item())

    res = { 
           "mAO"                : sum(AO)/len(AO), # mean average offset, interval width
           "mCP"                : sum(CP)/len(CP), # mean coverage percentage 
           "mCCE"               : sum(CCE)/len(CCE), # mean confidence calibration error 
           "eval_points"        : Conf,
           "offset_coeffs"      : OC_val,
           "coverage_percentage": CP,
           "average_offset"     : AO,
           "calibration_error"  : CCE
    }
    
    if verbose:
        logger = logging.getLogger("default")
        logger.info("Uncertainty Metrics")
        logger.info(" ".join(["{}: {:.3f},".format(k, v) for k, v in res.items() if isinstance(v, (int, float))]))
        logger.info("Evaluated confidence interval {}".format(Conf))
        logger.info("Corresponding data coverage percentage {} \n".format(np.round(CP, 2).tolist()))

    return res

def prediction_metrics(pred, label):
    
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