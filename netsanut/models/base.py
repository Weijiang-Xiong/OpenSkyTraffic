""" This file defines a base model for a interfaces and 
    shared functionalities
"""
from typing import Dict, Any

import torch 
import torch.nn as nn 

from netsanut.util import default_metrics
from netsanut.data import TensorDataScaler

class BaseModel(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.datascaler: TensorDataScaler
        self.metrics: dict
        
    @property   
    def device(self):
        return list(self.parameters())[0].device
    
    @property
    def num_params(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])
    
    
    def forward(self, data: dict[str, torch.Tensor]):
        """
            time series forecasting task, 
            data is assumed to have (N, T, M, C) shape (assumed to be unnormalized)
            label is assumed to have (N, T, M) shape (assumed to be unnormalized)
            
            compute loss in training mode, predict future values in inference
        """
        
        # preprocessing (if any)
        source = self.datascaler.transform(data['source'].to(self.device))
        target = data['target'].to(self.device)
        
        if self.training:
            assert target is not None, "label should be provided for training"
            return self.compute_loss(source, target)
        else:
            return self.inference(source, target)

    def make_pred(self, source: torch.Tensor) -> Any:
        """ This function will run a forward pass of the model with the source sequence
            the returned values are flexible, and may include anything depending on the 
            needs of `compute_loss` and `inference`
        """
        raise NotImplementedError

    def compute_loss(self, source: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """ This will be used in training, to compute the loss using source and target, and it
            can first call `make_pred` to have the prediction of the model, and then forward the loss
        """
        raise NotImplementedError
    
    def inference(self, source:torch.Tensor, target: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        """ this function also calls `make_pred` to obtain predictions, 
            if `target` is also available, it can also call `compute_auxiliary_metrics`
            different models may have different post-processing, for example some filtering
        """
        raise NotImplementedError
        
    def compute_auxiliary_metrics(self, pred, target) -> Dict[str, torch.Tensor]:
        
        # prevent gradient calculation when providing auxiliary metrics in training mode
        with torch.no_grad():
            metrics = default_metrics(pred, target)
            
        return {k:v for k, v in metrics.items()}
    
    
    def pop_auxiliary_metrics(self, scalar_only=True) -> Dict[str, float|torch.Tensor]:
        """ the auxiliary metrics will be recorded in dict `self.metrics`, this function
            pops the metrics and reset `self.metrics`. 
            
            will return scalar values if `scalar_only` is true, tensors will be ignored
        """
        if not self.record_auxiliary_metrics:
            return {} 
        
        if scalar_only:
            scalar_metrics = {k:v for k, v in self.metrics.items() if isinstance(v, (int, float))}
        else:
            scalar_metrics = self.metrics
            
        self.metrics = dict() 
        return scalar_metrics
    
    
    def visualize(self, data, label):
        raise NotImplementedError
    
    def adapt_to_metadata(self, metadata):
        raise NotImplementedError
    
