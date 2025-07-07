import torch
import logging
import numpy as np

logger = logging.getLogger("default")

def report_nan(pred:torch.tensor):
    num_of_nan = torch.sum(torch.isnan(pred)).item() 
    if num_of_nan > 0:
        logger.warning("number of nan in model predictions: {}".format(num_of_nan))

def MAE(pred, label, ignore_value=np.nan) -> float:
    # label can contain nan values to indicate no valid observations, 
    # but pred should not since it is the output of a neural network
    report_nan(pred)
    if np.isnan(ignore_value):
        mask = ~torch.isnan(label)
    else:
        mask = (label!=ignore_value)
    
    error = torch.abs(pred-label)
    error = error[mask]

    # when we have no valid labels (may happen in a batch or a specific location)
    # return 0 as no groundtruth to evaluate
    if error.numel() == 0:
        return 0

    return torch.mean(error).item()

def RMSE(pred, label, ignore_value=np.nan) -> float:
    report_nan(pred)
    if np.isnan(ignore_value):
        mask = ~torch.isnan(label)
    else:
        mask = (label!=ignore_value)
    
    error = (pred-label)**2
    error = error[mask]

    if error.numel() == 0:
        return np.nan

    return torch.sqrt(torch.mean(error)).item()

def MAPE(pred, label, ignore_value=np.nan, threshold=None) -> float:
    report_nan(pred)
    if np.isnan(ignore_value):
        mask = ~torch.isnan(label)
    else:
        mask = (label!=ignore_value)
    
    # we only keep the data points that are valid (not ignore_value) and larger than threshold
    if threshold is not None:
        mask = mask & (np.abs(label) > threshold)
    
    error = torch.abs((pred-label)/label)
    error = error[mask]

    if error.numel() == 0:
        return 0

    return torch.mean(error).item()

def common_metrics(pred, label, ignore_value=0.0, mape_threshold=None):
    
    mae = MAE(pred,label,ignore_value)
    mape = MAPE(pred,label,ignore_value, threshold=mape_threshold)
    rmse = RMSE(pred,label,ignore_value)
    
    return {"mae":mae, "mape":mape, "rmse":rmse}