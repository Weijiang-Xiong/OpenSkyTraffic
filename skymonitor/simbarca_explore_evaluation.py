import logging
from typing import Dict
from copy import deepcopy

import numpy as np

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
        convert_units: bool = False,
    ) -> None:
        super().__init__(save_dir, visualize, collect_pred, collect_data)
        self.save_dir 
        self.visualize
        self.collect_pred
        self.collect_data
        self.ignore_value = ignore_value
        self.eval_speed = eval_speed
        self.num_repeat = num_repeat
        # whether to convert the units of flow and density to standard units (vehicles per hour and vehicles per km)
        self.convert_units = convert_units

    def evaluate(self, model: nn.Module, dataloader: DataLoader, verbose: bool = False) -> Dict[str, float]:
        
        if self.num_repeat is None:
            self.num_repeat = 1
        else:
            logger.info(f"Evaluating {self.num_repeat} times and averaging the results.")

        metrics_of_repeats = []

        for _ in range(self.num_repeat):
            all_preds, all_labels = self.collect_predictions(model, dataloader, pred_seqs=self.collect_pred, data_seqs=self.collect_data)
            pred, label = all_preds['pred'], all_labels['target']

            data_channels = dataloader.dataset.data_channels['target']
            metrics = self.calculate_error_metrics(pred, label, data_channels, verbose=verbose)
            metrics_of_repeats.append(deepcopy(metrics))
        
        # average metrics over repeats, report std as well
        avg_metrics = {}
        for key in metrics_of_repeats[0].keys():
            results_over_repeats = np.array([m[key] for m in metrics_of_repeats])
            avg_metrics[key] = results_over_repeats.mean()
            avg_metrics[key + "_std"] = results_over_repeats.std()

        self.metrics_scalar.update(avg_metrics)
        logger.info(f"Evaluation results over {self.num_repeat} repeats: {avg_metrics}")

        return avg_metrics

    def calculate_error_metrics(self, pred: torch.Tensor, label: torch.Tensor, data_channels, verbose:bool=False):

        if self.convert_units:
            logger.info("Evaluating traffic variables using converted units (e.g., veh/min, veh/km and km/h).")
            # assuming the original units are vehicles per 5 minutes and vehicles per 100 meters
            factors = {"flow": 60, "density": 1000, "speed": 3.6}
        else:
            logger.info("Evaluating traffic variables using original units (e.g., veh/s, veh/m, m/s).")
            factors = {"flow": 1.0, "density": 1.0, "speed": 1.0}

        eval_res_by_horizon, avg_eval_res = [dict() for _ in range(len(data_channels))]

        for idx, channel in enumerate(data_channels):
            logger.info("Evaluating on variable {}".format(channel))
            channel_eval_res_by_horizon = self.common_metrics_by_horizon(
                factors[channel] * pred[..., idx], 
                factors[channel] * label[..., idx], 
                verbose=verbose)
            channel_avg_eval_res = self.average_metrics(channel_eval_res_by_horizon, verbose=verbose)
            eval_res_by_horizon[channel] = channel_eval_res_by_horizon
            avg_eval_res[channel] = channel_avg_eval_res
        
        if self.eval_speed:
            logger.info("Evaluating on derived variable speed, computed as flow / density")
            speed_labels = label[..., data_channels.index('flow')] / (label[..., data_channels.index('density')])
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
                
            speed_eval_res_by_horizon = self.common_metrics_by_horizon(
                factors['speed'] * speed_preds, 
                factors['speed'] * speed_labels, 
                verbose=verbose)
            speed_avg_eval_res = self.average_metrics(speed_eval_res_by_horizon, verbose=verbose)
            eval_res_by_horizon['speed'] = speed_eval_res_by_horizon
            avg_eval_res['speed'] = speed_avg_eval_res
        
        eval_res_by_horizon = flatten_results_dict(eval_res_by_horizon)
        avg_eval_res = flatten_results_dict(avg_eval_res)

        self.metrics_scalar.update(avg_eval_res)
        self.metrics_vector.update(eval_res_by_horizon)

        return deepcopy(self.metrics_scalar)