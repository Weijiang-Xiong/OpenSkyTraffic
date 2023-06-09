import torch
from typing import List

class TensorDataScaler:
    """
    normalize the data, a simplified version of sklearn.preprocessing.StandardScaler
    """

    def __init__(self, mean, std, data_dim:int=0):
            
        self.data_dim = data_dim
        self.mean = mean[data_dim]
        self.std = std[data_dim]
        self.inv_std = 1.0 / self.std
        
    def transform(self, data):
        
        if data.dim() == 4: # assume N, T, M, C
            data[..., self.data_dim] = (data[..., self.data_dim] - self.mean) * self.inv_std
        elif data.dim() == 3: # (N, T, M) or (N, M, T), in case of data dimension C=1
            data = (data - self.mean) * self.inv_std
            
        return data

    def inverse_transform(self, data):
        
        if len(data.shape) == 4: # assume N, T, M, C
            data[..., self.data_dim] = (data[..., self.data_dim] * self.std) + self.mean
        elif len(data.shape) == 3: # (N, T, M) or (N, M, T), in case of data dimension C=1
            data = (data * self.std) + self.mean
            
        return data

    def inverse_transform_logvar(self, logvar):
        
        """ 
            we scale the raw input x to zero mean and unit variance variable u using self.transform:
                u = (x - mean) / std 
            when we scale back, 
                x = u * std + mean 
            the variance should be
                Var(x) = Var(u) * std**2
            and the log variance should be:
                log Var(x) = log(Var(u)) + 2log(std)
        """
        return logvar + 2*torch.log(self.std)
