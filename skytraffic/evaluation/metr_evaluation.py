import json
import logging
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base_evaluator import BaseEvaluator
from .metrics import common_metrics
from ..utils.io import make_dir_if_not_exist

logger = logging.getLogger("default")

class MetrEvaluator(BaseEvaluator):

    ignore_value = 0.0
    mape_threshold = None

    def __init__(
        self,
        save_dir: str = None,
        visualize: bool = False,
        collect_pred=["pred"],
        collect_data=["target"],
    ) -> None:
        super().__init__(save_dir, visualize, collect_pred, collect_data)
        self.save_dir 
        self.visualize
        self.collect_pred
        self.collect_data

    def evaluate(self, model: nn.Module, dataloader: DataLoader, verbose: bool = False) -> Dict[str, float]:

        all_preds, all_labels = self.collect_predictions(model, dataloader, pred_seqs=self.collect_pred, data_seqs=self.collect_data)
        pred, label = all_preds['pred'], all_labels['target']

        # evaluate each predicted time step, i.e., forecasting from 5 min up to 1 hour
        eval_res_by_horizon = self.common_metrics_by_horizon(pred, label, verbose=verbose)
        avg_eval_res = self.average_metrics(eval_res_by_horizon, verbose=verbose)

        self.metrics_scalar.update(avg_eval_res)
        self.metrics_vector.update(eval_res_by_horizon)

        return self.metrics_scalar
