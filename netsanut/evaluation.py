import logging
from typing import Dict

import numpy as np 
from scipy.stats import gennorm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from netsanut.util import default_metrics
from collections import defaultdict

def GGD_interval(beta:int, confidence):
    """ 
        compute a `k` for a Generalized Gaussian Distribution parameterized by `beta`
        such that 0.5 * confidence_interval_width = k * std, 
        one can later obtain the confidence interval by
            upper_bound = mean + k * pred_std
            lower_bound = mean - k * pred_std
            
        References
            https://en.wikipedia.org/wiki/Generalized_normal_distribution
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gennorm.html
    """
    
    rv = gennorm(beta=beta)
    k:float = abs(rv.interval(confidence)[0])
    
    return k

def uncertainty_metrics(pred: torch.Tensor, target: torch.Tensor, scale:torch.Tensor, 
                        loss_exponent:int=2, ignore_value=0.0, verbose=False):
    pred, target, scale = pred.flatten(), target.flatten(), scale.flatten()
    if ignore_value is not None:
        valid = (target != ignore_value)
        pred, target, scale = pred[valid], target[valid], scale[valid]
    
    eval_points = np.round(np.arange(0.5, 1.0, 0.05), 2).tolist()
    
    all_cp, CCE, AO = [], [], []
    for confidence in eval_points:
        k = GGD_interval(beta=loss_exponent, confidence=confidence)
        ub = pred + k * scale
        lb = pred - k * scale
        
        covered = torch.logical_and(target < ub, target > lb)
        coverage_percentage = covered.sum() / covered.numel()
        all_cp.append(coverage_percentage.item())
        CCE.append((confidence - coverage_percentage).item())
        AO.append(torch.mean(k*scale).item())

    res = { 
           "mAO"                : sum(AO)/len(AO), # mean average offset, interval width
           "mCP"                : sum(all_cp)/len(all_cp), # mean coverage percentage 
           "mCCE"               : sum(CCE)/len(CCE), # mean confidence calibration error 
           "eval_points"        : eval_points,
           "coverage_percentage": all_cp,
           "average_offset"     : AO,
           "calibration_error"  : CCE
    }
    
    if verbose:
        logger = logging.getLogger("default")
        logger.info("Uncertainty Metrics")
        logger.info(" ".join(["{}: {},".format(k, v) for k, v in res.items() if isinstance(v, (int, float))]))
        logger.info("Evaluated confidence interval {}".format(eval_points))
        logger.info("Corresponding data coverage percentage {}".format(np.round(all_cp, 2).tolist()))

    return res

def inference_on_dataset(model:nn.Module, dataloader:DataLoader) -> Dict[str, torch.Tensor]:
    """ run inference of the model on the dataloader
        concatenate all predictions and corresponding labels.
        
        Returns: predictions, labels
    """
    model.eval()

    all_res = defaultdict(list)
    for data in dataloader:
        result_dict = model(data)
        for dictionary in [data, result_dict]:
            for key, value in dictionary.items():
                all_res[key].append(value)

    for key, value in all_res.items():
        all_res[key] = torch.cat(value, dim=0).detach().cpu()
    
    return all_res

def evaluate(model: nn.Module, dataloader: DataLoader, 
             verbose=False, eval_uncertainty=False) -> Dict[str, float]:

    logger = logging.getLogger("default")
    
    all_res = inference_on_dataset(model, dataloader)
    all_preds, all_labels = all_res['pred'], all_res['target']

    if verbose:
        logger.info("The shape of predicted {} and label {}".format(all_preds.shape, all_labels.shape))

    for i in range(12):  # number of predicted time step
        pred = all_preds[:, i, :]
        real = all_labels[:, i, :]
        aux_metrics = default_metrics(pred, real)

        if verbose:
            logger.info('Evaluate model on test data at {:d} time step'.format(i+1))
            logger.info('Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'.format(
                aux_metrics['mae'], aux_metrics['mape'], aux_metrics['rmse']
            )
            )

    res = default_metrics(all_preds, all_labels)
    if verbose:
        logger.info('On average over 12 different time steps')
        logger.info('Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'.format(
            res['mae'], res['mape'], res['rmse']
        )
        )

    if eval_uncertainty:
        all_scales = torch.pow(all_res['logvar'].exp(), exponent=model.loss.exponent)
        res_u = uncertainty_metrics(all_preds, all_labels, all_scales, model.loss.exponent, verbose=verbose)
        res.update(res_u)

    return res