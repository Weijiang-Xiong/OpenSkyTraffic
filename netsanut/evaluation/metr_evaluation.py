import logging
from typing import Dict

import numpy as np 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import prediction_metrics, uncertainty_metrics
from collections import defaultdict

EVAL_CONFS = np.round(np.arange(0.5, 1.0, 0.05), 2).tolist()

class MetrEvaluator:
    
    def __init__(self, save_dir: str=None, save_res: bool=True, save_note:str=None) -> None:
        self.save_dir = save_dir
        self.save_res = save_res
        self.save_note = save_note if save_note is not None else "default"
    
    def __call__(self, model: nn.Module, dataloader: DataLoader, **kwargs) -> Dict[str, float]:
        return self.evaluate(model, dataloader, **kwargs)
    
    def collect_predictions(self, model:nn.Module, dataloader:DataLoader) -> Dict[str, torch.Tensor]:
        """ run inference of the model on the dataloader
            concatenate all predictions and corresponding labels.
            
            Returns: predictions, labels
        """
        model.eval()

        all_res = defaultdict(list)
        for data in dataloader:
            with torch.no_grad():
                result_dict = model(data)
            for dictionary in [data, result_dict]:
                for key, value in dictionary.items():
                    all_res[key].append(value)

        for key, value in all_res.items():
            all_res[key] = torch.cat(value, dim=0).detach().cpu()
        
        return all_res

    def evaluate(self, model: nn.Module, dataloader: DataLoader, verbose=False) -> Dict[str, float]:

        logger = logging.getLogger("default")
        
        all_res = self.collect_predictions(model, dataloader)
        all_preds, all_labels = all_res['pred'], all_res['target']

        if verbose:
            logger.info("The shape of predicted {} and label {}".format(all_preds.shape, all_labels.shape))

        # evaluate each predicted time step, i.e., forecasting from 5 min up to 1 hour
        for i in range(12):  # number of predicted time step
            pred = all_preds[:, i, :]
            real = all_labels[:, i, :]
            step_metrics = prediction_metrics(pred, real)

            if verbose:
                logger.info('Evaluate model on test data at {:d} time step'.format(i+1))
                logger.info('MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}'.format(
                    step_metrics['mae'], step_metrics['mape'], step_metrics['rmse']
                )
                )

        # average performance on all 12 prediction steps, usually not reported in papers
        res = prediction_metrics(all_preds, all_labels)
        if verbose:
            logger.info('On average over 12 different time steps')
            logger.info('MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}'.format(
                res['mae'], res['mape'], res['rmse']
            )
            )

        # evaluate uncertainty prediction if possible
        if getattr(model, 'is_probabilistic', False):
            all_scales = all_res['sigma']
            offset_coeffs = {c:model.offset_coeff(confidence=c) for c in EVAL_CONFS}
            res_u = uncertainty_metrics(all_preds, all_labels, all_scales, offset_coeffs, verbose=verbose)
            res.update(res_u)

        return res