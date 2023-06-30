import logging
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from netsanut.util import default_metrics


def inference_on_dataset(model:nn.Module, dataloader:DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """ run inference of the model on the dataloader
        concatenate all predictions and corresponding labels.
        
        Returns: predictions, labels
    """
    model.eval()

    all_preds, all_labels = [], []
    for data, label in dataloader:

        data, label = data.cuda(), label.cuda()

        preds = model(data)

        all_preds.append(preds)
        all_labels.append(label)

    all_preds = torch.cat(all_preds, dim=0).cpu()
    all_labels = torch.cat(all_labels, dim=0).cpu()
    
    return all_preds, all_labels

def evaluate(model: nn.Module, dataloader: DataLoader, verbose=False):

    logger = logging.getLogger("default")
    
    all_preds, all_labels = inference_on_dataset(model, dataloader)

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