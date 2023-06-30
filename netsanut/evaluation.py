import logging
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from netsanut.util import default_metrics
from collections import defaultdict

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

def evaluate(model: nn.Module, dataloader: DataLoader, verbose=False) -> Dict[str, float]:

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

    overall_metrics = default_metrics(all_preds, all_labels)

    if verbose:
        logger.info('On average over 12 different time steps')
        logger.info('Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'.format(
            overall_metrics['mae'], overall_metrics['mape'], overall_metrics['rmse']
        )
        )

    return overall_metrics