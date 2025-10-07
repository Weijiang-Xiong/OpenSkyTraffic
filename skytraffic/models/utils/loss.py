import logging

import torch 
import torch.nn as nn 

import numpy as np 

logger = logging.getLogger("default")
class GeneralizedProbRegLoss(nn.Module):
    
    """ 
        This class implements a generalized probabilistic regression loss. 
        If aleatoric is set to False, the loss will be a Lp regression loss:

            loss = (pred - label) ** exponent 
        
        in this case, the variance prediction branch will not be trained, they receive no gradients. 
        
        If aleatoric is set to True, the regression loss will be attenuated by a predicted variance:

            loss = (pred - label) ** exponent / exp(plog_sigma) + alpha * plog_sigma
        
        Specifically, if aleatoric = True, exponent=2 and alpha=1, the loss will be 
        the Gaussian Negative Log Likelihood Loss, as described in
        https://proceedings.neurips.cc/paper/2017/file/2650d6089a6d640c5e85b2b88265dc2b-Paper.pdf
        
        Here we allow exponent to be a positive integer, instead of fixing it to 2 for the Gaussian case.
        Therefore we should generalize log-variance, i.e., log(sigma^2) to log(sigma^exponent), which
        turns out to be p*log(sigma) with p being the exponent.
    """
    
    def __init__(self, reduction="mean", aleatoric=False, exponent=1, alpha=1.0, ignore_value:float=float("nan")):
        self.aleatoric = aleatoric # use aleatoric uncertainty to attenuate the regression loss
        self.exponent = exponent 
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_value = ignore_value # the value corresponding to "no reading" or "not valid"
        super(self.__class__, self).__init__()
    
    def forward(self, pred, label, plog_sigma=None):
        """
        pred: (N, T, M)
        label: (N, T, M)
        plog_sigma: (N, T, M), p*log(sigma) as indicated in Gaussian NLL Loss above. It means the scale of uncertainty.
        """
        
        loss = torch.pow(torch.abs(pred-label), self.exponent)
        
        if self.aleatoric and plog_sigma is not None:
            loss = torch.multiply(loss, torch.exp(-plog_sigma)) + self.alpha * plog_sigma
        
        # the masking part is modified from DCRNN 
        # https://github.com/liyaguang/DCRNN/blob/master/lib/metrics.py#L75
        if self.ignore_value is not None:
            if np.isnan(self.ignore_value): # == and != do not work with nan values
                mask = (~torch.isnan(label)).type(torch.float32)
            else:
                mask = (label != self.ignore_value).type(torch.float32)
            mask /= torch.mean(mask)
            mask = self.nan_to_num(mask)
            loss *= mask

        loss = self.nan_to_num(loss)
        
        match self.reduction:
            case "mean":
                loss = torch.mean(loss) 
            case "sum":
                loss = torch.sum(loss) 
            case _:
                pass 
            
        return loss
    
    @staticmethod
    def nan_to_num(tensor):
        return torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    
    # this appears in `print`
    def extra_repr(self) -> str:
        return "aleatoric={}, exponent={}, alpha={}, ignore_value={}".format(
            self.aleatoric, self.exponent, self.alpha, self.ignore_value)

""" Copied from https://github.com/nnzhan/Graph-WaveNet/blob/master/util.py
"""

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


if __name__ == "__main__":
    print(GeneralizedProbRegLoss())
