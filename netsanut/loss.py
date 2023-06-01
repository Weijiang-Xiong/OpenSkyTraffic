import logging

import torch 
import torch.nn as nn 

logger = logging.getLogger("default")
class GeneralizedProbRegLoss(nn.Module):
    
    """ 
        This class implements a generalized probabilistic regression loss. 
        If aleatoric is set to False, the loss will be a Lp regression loss:

            loss = (pred - label) ** exponent 
        
        in this case, the variance prediction branch will not be trained, they receive no gradients. 
        
        If aleatoric is set to True, the regression loss will be attenuated by a predicted variance:

            loss = (pred - label) ** exponent / exp(logvar) + alpha * logvar
        
        Specifically, if aleatoric = True, exponent=2 and alpha=1, the loss will be 
        the Gaussian Negative Log Likelihood Loss, as described in
        https://proceedings.neurips.cc/paper/2017/file/2650d6089a6d640c5e85b2b88265dc2b-Paper.pdf
    """
    
    def __init__(self, reduction="mean", aleatoric=False, exponent=1, alpha=1.0, ignore_value:float=0.0):
        self.aleatoric = aleatoric # use aleatoric uncertainty to attenuate the regression loss
        self.exponent = exponent 
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_value = ignore_value # the value corresponding to "no reading" or "not valid"
        super(self.__class__, self).__init__()
    
    def forward(self, pred, label, logvar=None):
        """
        pred: (N, T, M)
        label: (N, T, M)
        logvar: (N, T, M), log variance
        but the shape doesn't really matter, as long as they match 
        """
        
        loss = torch.pow(torch.abs(pred-label), self.exponent)
        
        if self.aleatoric:
            if logvar is not None:
                loss = torch.multiply(loss, torch.exp(-logvar))  + self.alpha * logvar
            else:
                logger.warning("Did not receive Log Variance for loss computation, skip aleatoric part")
        
        loss = self.nan_to_num(loss)
        
        if self.ignore_value is not None:
            mask = (label != self.ignore_value).type(torch.float32)
            mask /= torch.mean(mask)
            mask = self.nan_to_num(mask)
            loss *= mask
        
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
        
        if torch.any(torch.isnan(tensor)):
            logger.warning("Encountered Nan, replacing them with 0, but training could collapse")
            tensor = torch.nan_to_num(tensor, nan=0.0)
            
        return tensor 
    

def test_forward():
    rand_output = torch.rand(size=(5,5), requires_grad=True)
    rand_label  = torch.rand_like(rand_output)
    rand_logvar = torch.rand_like(rand_output, requires_grad=True)
    
    criterion = GeneralizedProbRegLoss(reduction='mean', aleatoric=True)
    loss = criterion(rand_output, rand_label, rand_logvar)
    loss.backward() 
    
    assert rand_output.grad is not None
    assert rand_logvar.grad is not None

def test_forward_without_logvar():
    rand_output = torch.rand(size=(5,5), requires_grad=True)
    rand_label  = torch.rand_like(rand_output)
    rand_logvar = torch.rand_like(rand_output, requires_grad=True)
    
    criterion = GeneralizedProbRegLoss(reduction='mean', aleatoric=True)
    loss = criterion(rand_output, rand_label)
    loss.backward() 
    
    assert rand_output.grad is not None
    assert rand_logvar.grad is None
    
def test_ignore_value():
    rand_output = torch.ones(size=(5,5), requires_grad=True)
    rand_label  = torch.ones_like(rand_output)
    rand_label[0, ...] = 0.0
    
    criterion = GeneralizedProbRegLoss(reduction=None, aleatoric=False, ignore_value=0.0)
    loss = criterion(rand_output, rand_label)
    assert torch.allclose(loss, torch.tensor(0.0))

def test_nan_to_num():
    rand_output = torch.ones(size=(5,5), requires_grad=True)
    rand_label  = torch.zeros_like(rand_output)
    
    criterion = GeneralizedProbRegLoss(reduction="mean", aleatoric=False, ignore_value=0.0)
    loss = criterion(rand_output, rand_label)
    assert torch.allclose(loss, torch.tensor(0.0))
    
if __name__ == "__main__":
    test_forward()
    test_forward_without_logvar()
    test_ignore_value()
    test_nan_to_num()
    
