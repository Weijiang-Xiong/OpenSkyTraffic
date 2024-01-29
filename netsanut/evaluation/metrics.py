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

def masked_mape_with_threshold(preds, labels, null_val=np.nan, threshold=10):
    less_than_threshold = labels < threshold
    preds, labels = preds.clone(), labels.clone()
    preds[less_than_threshold] = null_val
    labels[less_than_threshold] = null_val
    return masked_mape(preds, labels, null_val)

def uncertainty_metrics(pred: torch.Tensor, target: torch.Tensor, scale:torch.Tensor, 
                        offset_coeffs: Dict[float, float], ignore_value=0.0, verbose=False):
    """ The uncertainty prediction is evaluated by the corresponding confidence intervals, where we assume the interval is centered on the expected value, and the upper bound and lower bound 
    are expressed by multiples of the predicted uncertainty scale. 
    
    The primary metrics are 
        1. mAO, the half-width of the confidence intervals averaged over all confidences and all predictions. 
        2. mCP, the percentage of data points covered by the confidence interval, averaged over all confidences 
        3. mCCE, the difference between confidence and data coverage, averaged over all confidences
    
    A calibrated model is expected to predict confidence intervals whose coverage is the same as the confidence score. That will result in a zero mCCE. 
    
    Args:
        pred (torch.Tensor): the expected future value predicted by the model
        target (torch.Tensor): the real future value from data
        scale (torch.Tensor): the predicted uncertainty scale, have the same unit as `pred`
        offset_coeffs (Dict[float, float]): pairs of confidence score and interval offsets
        ignore_value (float, optional): the values corresponding to "no data". Defaults to 0.0.
        verbose (bool, optional): whether to print evaluation results. Defaults to False.

    Returns:
        res: collection of evaluation results
    """
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

def prediction_metrics(pred, label, ignore_value=0.0, mape_threshold=0.0):
    
    mae = masked_mae(pred,label,ignore_value).item()
    mape = masked_mape_with_threshold(pred,label,ignore_value, threshold=mape_threshold).item()
    rmse = masked_rmse(pred,label,ignore_value).item()
    
    return {"mae":mae, "mape":mape, "rmse":rmse}


def report_nan(res:torch.tensor, use_logger=True):
    logger = logging.getLogger("default")
    num_of_nan = torch.sum(torch.isnan(res)).item()
    if num_of_nan > 0:
        if not use_logger:
            print("number of nan in result: {}".format(torch.sum(torch.isnan(res)).item()))
        else:
            logger.warning("number of nan in result: {}".format(torch.sum(torch.isnan(res)).item()))