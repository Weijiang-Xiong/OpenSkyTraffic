""" This file defines a base model for a interfaces and 
    shared functionalities
"""

import torch 
import torch.nn as nn 

from netsanut.util import default_metrics



class BaseModel(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    @property   
    def device(self):
        return list(self.parameters())[0].device
    
    def compute_auxiliary_metrics(self, pred, label):
        
        # prevent gradient calculation when providing auxiliary metrics in training mode
        with torch.no_grad():
            metrics = default_metrics(pred, label)
            
        return {k:v for k, v in metrics.items()}
    
    
    def pop_auxiliary_metrics(self, scalar_only=True):
        """ auxiliary metrics may also contain the log variance, 
            so use scalar_only to control the behavior
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
    
