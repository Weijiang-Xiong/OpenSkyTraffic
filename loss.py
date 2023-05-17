import torch 
import torch.nn as nn 

class GaussianNLLLoss(nn.Module):
    
    """ Gaussian Negative Log Likelihood Loss, as described in
        https://proceedings.neurips.cc/paper/2017/file/2650d6089a6d640c5e85b2b88265dc2b-Paper.pdf
    """
    
    def __init__(self, reduction="mean"):
        self.reduction = reduction
        super(GaussianNLLLoss, self).__init__()
    
    def forward(self, pred, label, logvar):
        """
        pred: (N, T, M)
        label: (N, T, M)
        logvar: (N, T, M), log variance
        but the shape doesn't really matter, as long as they match 
        """
        
        loss = (pred - label)**2 * torch.exp(-logvar) + logvar
        
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
            
        return loss