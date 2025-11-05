import logging
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from skytraffic.evaluation import BaseEvaluator
from skytraffic.utils.io import flatten_results_dict

logger = logging.getLogger("default")

class SimBarcaExploreEvaluator(BaseEvaluator):

    ignore_value = 0.0
    mape_threshold = 0.1

    def __init__(
        self,
        save_dir: str = None,
        visualize: bool = False,
        collect_pred=["pred"],
        collect_data=["target"],
        ignore_value: float = 0.0,
        eval_speed: bool = False,
        num_repeat: int = None,
    ) -> None:
        super().__init__(save_dir, visualize, collect_pred, collect_data)
        self.save_dir 
        self.visualize
        self.collect_pred
        self.collect_data
        self.ignore_value = ignore_value
        self.eval_speed = eval_speed
        self.num_repeat = num_repeat

    def evaluate(self, model: nn.Module, dataloader: DataLoader, verbose: bool = False) -> Dict[str, float]:

        all_preds, all_labels = self.collect_predictions(model, dataloader, pred_seqs=self.collect_pred, data_seqs=self.collect_data)
        pred, label = all_preds['pred'], all_labels['target']

        data_channels = dataloader.dataset.data_channels['target']
        eval_res_by_horizon, avg_eval_res = [dict() for _ in range(len(data_channels))]

        for idx, channel in enumerate(data_channels):
            logger.info("Evaluating on variable {}".format(channel))
            channel_eval_res_by_horizon = self.common_metrics_by_horizon(pred[..., idx], label[..., idx], verbose=verbose)
            channel_avg_eval_res = self.average_metrics(channel_eval_res_by_horizon, verbose=verbose)
            eval_res_by_horizon[channel] = channel_eval_res_by_horizon
            avg_eval_res[channel] = channel_avg_eval_res
        
        if self.eval_speed:
            logger.info("Evaluating on derived variable speed, computed as flow / density")
            speed_labels = all_labels[..., data_channels.index('flow')] / (all_labels[..., data_channels.index('density')])
            # mark invalid speed values with self.ignore_value
            speed_labels = torch.where(
                torch.isfinite(speed_labels), 
                speed_labels, 
                torch.full_like(speed_labels, self.ignore_value)
            )
            if pred.shape[-1] == 3: # the model predict speed directly, and thus the channel dimension is 3 instead of 2
                logger.info("Model predicts speed directly.")
                speed_preds = pred[..., -1]
            else:
                logger.info("Model does not predict speed directly, computing speed using predicted flow and density.")
                speed_preds = pred[..., data_channels.index('flow')] / (pred[..., data_channels.index('density')] + 1e-6)
                
            speed_eval_res_by_horizon = self.common_metrics_by_horizon(speed_preds, speed_labels, verbose=verbose)
            speed_avg_eval_res = self.average_metrics(speed_eval_res_by_horizon, verbose=verbose)
            eval_res_by_horizon['speed'] = speed_eval_res_by_horizon
            avg_eval_res['speed'] = speed_avg_eval_res
        
        eval_res_by_horizon = flatten_results_dict(eval_res_by_horizon)
        avg_eval_res = flatten_results_dict(avg_eval_res)

        self.metrics_scalar.update(avg_eval_res)
        self.metrics_vector.update(eval_res_by_horizon)

        return self.metrics_scalar