import logging
from typing import Dict

import numpy as np 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import common_metrics, gaussian_dist_metrics
from collections import defaultdict

EVAL_CONFS = np.round(np.arange(0.5, 1.0, 0.05), 2).tolist()

class MetrEvaluator:
    
    def __init__(self, save_dir: str=None, save_note:str=None, mape_threshold:float=0.0, visualize:bool=False) -> None:
        self.save_dir = save_dir
        self.save_note = save_note if save_note is not None else "default"
        self.visualize = visualize
        self.mape_threshold = mape_threshold
    
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

        # evaluate each predicted time step, i.e., forecasting from 5 min up to 1 hour
        for i in range(12):  # number of predicted time step
            pred = all_preds[:, i, :]
            real = all_labels[:, i, :]
            step_res = common_metrics(pred, real, mape_threshold=self.mape_threshold)
            if verbose:
                logger.info('Evaluate model on test data at {:d} time step'.format(i+1))
                logger.info(
                    'MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}'.format(step_res['mae'], step_res['mape'], step_res['rmse'])
                )

        # average performance on all 12 prediction steps, usually not reported in papers
        res = common_metrics(all_preds, all_labels, mape_threshold=self.mape_threshold)
        if verbose:
            logger.info('On average over 12 different time steps')
            logger.info(
                'MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}'.format(res['mae'], res['mape'], res['rmse'])
                )

        return res