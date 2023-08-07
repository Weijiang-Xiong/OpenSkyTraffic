import logging
from typing import Dict

import numpy as np 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from netsanut.util import default_metrics
from collections import defaultdict

EVAL_CONFS = np.round(np.arange(0.5, 1.0, 0.05), 2).tolist()

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
        step_metrics = default_metrics(pred, real)

        if verbose:
            logger.info('Evaluate model on test data at {:d} time step'.format(i+1))
            logger.info('MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}'.format(
                step_metrics['mae'], step_metrics['mape'], step_metrics['rmse']
            )
            )

    res = default_metrics(all_preds, all_labels)
    if verbose:
        logger.info('On average over 12 different time steps')
        logger.info('MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}'.format(
            res['mae'], res['mape'], res['rmse']
        )
        )

    if eval_uncertainty:
        all_scales = all_res['scale_u']
        offset_coeffs = {c:model.offset_coeff(confidence=c) for c in EVAL_CONFS}
        res_u = uncertainty_metrics(all_preds, all_labels, all_scales, offset_coeffs, verbose=verbose)
        res.update(res_u)

    return res