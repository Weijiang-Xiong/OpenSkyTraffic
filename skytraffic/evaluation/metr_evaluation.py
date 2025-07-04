import json
import logging
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import common_metrics
from ..utils.io import make_dir_if_not_exist

logger = logging.getLogger("default")

class MetrEvaluator:

    def __init__(
        self,
        save_dir: str = None,
        visualize: bool = False,
        collect_pred=["pred"],
        collect_data=["target"],
    ) -> None:
        self.save_dir = save_dir
        make_dir_if_not_exist(self.save_dir)
        self.visualize = visualize
        self.collect_pred = collect_pred
        self.collect_data = collect_data
        self.metrics_scalar: Dict[str, float] = dict()
        self.metrics_vector: Dict[str, List] = dict()

    def __call__(self, model: nn.Module, dataloader: DataLoader, **kwargs) -> Dict[str, float]:
        eval_res = self.evaluate(model, dataloader, **kwargs)

        if self.visualize:
            self.save_scores_to_json()

        return eval_res
    
    def collect_predictions(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        pred_seqs: List = None,
        data_seqs: List = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """run inference of the model on the data_loader concatenate all predictions and corresponding labels.
        
        The sequences in `data_seqs` will be collected from the data batch. 
        The sequences in `pred_seqs` will be collected from the model output. 
        Usually they are the same, but can be different when evaluation requires additional sequences.
        """
        model.eval()

        all_preds = defaultdict(list)
        all_labels = defaultdict(list)

        for data_dict in data_loader:
            with torch.no_grad():
                pred_dict = model(data_dict)
            
            # collect predicted sequences corresponding labels
            for name in pred_seqs:
                all_preds[name].append(pred_dict[name])
            for name in data_seqs: 
                all_labels[name].append(data_dict[name])
                
        # this will actually modify all_preds and all_labels
        for res_collection in [all_preds, all_labels]:
            for key, value in res_collection.items():
                res_collection[key] = torch.cat(value, dim=0).detach().cpu()
        
        return all_preds, all_labels

    def evaluate(self, model: nn.Module, dataloader: DataLoader, verbose: bool = False) -> Dict[str, float]:

        all_preds, all_labels = self.collect_predictions(model, dataloader, pred_seqs=self.collect_pred, data_seqs=self.collect_data)
        pred, label = all_preds['pred'], all_labels['target']

        # evaluate each predicted time step, i.e., forecasting from 5 min up to 1 hour
        eval_res_by_horizon = self.error_metrics_by_horizon(pred, label, verbose=verbose)
        avg_eval_res = self.average_error_metrics(eval_res_by_horizon, verbose=verbose)

        self.metrics_scalar.update(avg_eval_res)
        self.metrics_vector.update(eval_res_by_horizon)

        return self.metrics_scalar

    def error_metrics_by_horizon(self, pred, label, verbose: bool = False):
        eval_res_over_time = defaultdict(list)
        pred_steps = pred.shape[1]
        for i in range(pred_steps):  # number of predicted time step
            pred_i = pred[:, i, :]
            real_i = label[:, i, :]
            step_res = common_metrics(pred_i, real_i)

            if verbose:
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in step_res.items()])
                logger.info(f"Step {i} results: {metrics_str}")

            for k, v in step_res.items():
                eval_res_over_time[k].append(v)

        return eval_res_over_time
    

    def average_error_metrics(self, eval_res_by_horizon, verbose: bool = False):
        avg_eval_res = {k:sum(v)/len(v) for k, v in eval_res_by_horizon.items()}
        if verbose:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in avg_eval_res.items()])
            logger.info(f"Average results: {metrics_str}")
        return avg_eval_res
    

    def save_scores_to_json(self, file_name: str = "final_evaluation_scores.json"):
        """
        Save the scores to a JSON file.
        The scores are saved in a dictionary with keys being the score types and values being the scores.
        """
        scalar_res = {k:float(v) for k, v in self.metrics_scalar.items()}
        vector_res = {k:v for k, v in self.metrics_vector.items() if isinstance(v, list)}
        res_to_save = {
            "average": scalar_res,
            "horizon": vector_res
        }

        save_path = f"{self.save_dir}/{file_name}"
        with open(save_path, 'w') as f:
            json.dump(res_to_save, f, indent=4)
        logger.info(f"Saved scores to {save_path}")