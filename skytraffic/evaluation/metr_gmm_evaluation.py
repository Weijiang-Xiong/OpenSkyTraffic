import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict

from .metr_evaluation import MetrEvaluator

class MetrGMMEvaluator(MetrEvaluator):

    def __init__(self, save_dir: str = None, visualize: bool = False, collect_pred=["pred", "mixing", "means", "log_var"], collect_data=["target"]):
        super().__init__(save_dir, visualize, collect_pred, collect_data)

    def evaluate(self, model: nn.Module, dataloader: DataLoader, verbose: bool = False) -> Dict[str, float]:
        return super().evaluate(model, dataloader, verbose=verbose)