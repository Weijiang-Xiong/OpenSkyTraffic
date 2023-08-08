""" This file defines a base model for a interfaces and 
    shared functionalities
"""
import logging 

from typing import Dict, Any, Mapping, Tuple
from scipy.stats import rv_continuous, gennorm

import torch 
import torch.nn as nn 

from netsanut.data import TensorDataScaler

logger = logging.getLogger("default")

class BaseModel(nn.Module):
    
    def __init__(self, beta:int=None) -> None:
        super().__init__()
        self.datascaler: TensorDataScaler
        self._calibrated_intervals: Dict[float, float] = dict() 
        self._beta: int = beta
        self._distribution: rv_continuous = None
        self._set_distribution(beta=self._beta)

    @property   
    def device(self):
        return list(self.parameters())[0].device
    
    @property
    def num_params(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])
    
    @property
    def is_probabilistic(self):
        return self._distribution is not None

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
        """ this function calls `make_pred` to obtain predictions, 
            different models may have different post-processing, for example some filtering
        """
        raise NotImplementedError

    
    def adapt_to_metadata(self, metadata):
        raise NotImplementedError


    def _set_distribution(self, beta:int) -> rv_continuous:
        
        if beta is None:
            self._distribution = None 
        else:
            self._distribution = gennorm(beta=int(beta))

        return self._distribution
    
    def offset_coeff(self, 
                    confidence:float, 
                    return_calibrated=True
        )-> Dict[float, float]:
        """ 
        compute a `k` for a Generalized Gaussian Distribution parameterized by `beta`
        such that 0.5 * confidence_interval_width = k * std, 
        one can later obtain the confidence interval by
            upper_bound = mean + k * pred_std
            lower_bound = mean - k * pred_std
            
        References
            https://en.wikipedia.org/wiki/Generalized_normal_distribution
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gennorm.html

        Args:
            confidence (float): _description_
            return_calibrated (bool, optional): _description_. Defaults to True.

        Returns:
            Dict[float,Tuple[float, float]]: a mapping from confidence score to offset coefficient
        """
        # use the calibrated result if it has been calibrated 
        if return_calibrated and confidence in self._calibrated_intervals.keys():
            interval = self._calibrated_intervals[confidence]
        else:
            try:
                interval = self._distribution.interval(confidence)
            except AttributeError:
                logger.warning("Make sure self._distribution is properly initialized, and it has the method `interval` for confidence intervals")
                interval = (0.0, 0.0)
            
        return abs(interval[0])
    
    def post_process(self, res):
        # the scale of uncertainty, supposed to have the same unit as prediction
        # so it makes sense to add them. 
        res['scale_u'] = torch.exp(res['logvar']*(1.0/self._beta))
        return res
    
    def update_calibration(self, intervals: Dict[float,Tuple[float, float]]):
        self._calibrated_intervals.update(intervals)
    
    def state_dict(self):
        ret = super().state_dict()
        ret["_beta"] = self._beta
        ret["_calibrated_intervals"] = self._calibrated_intervals
        return ret
        
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = False):
        self._beta = state_dict["_beta"]
        self._set_distribution(self._beta)
        self._calibrated_intervals = state_dict["_calibrated_intervals"]
        del state_dict["_beta"]
        del state_dict["_calibrated_intervals"]
        super().load_state_dict(state_dict, strict)
        return