import torch
from typing import List

class TensorDataScaler:
    """
    normalize the data, a simplified version of sklearn.preprocessing.StandardScaler
    assume the data to have shape
        1. (N, T, P, C) where the 0 in the C dimension is the data, and the rest may be time in day, day in week
        2. (N, T, P) just data, no time appended.

    """

    def __init__(self, mean: float, std: float, data_dim: int = 0, device: str = "cuda"):
        self.data_dim = data_dim
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.inv_std = 1.0 / self.std
        self.to(device)

    def transform(self, data):
        if data.dim() == 4:  # assume N, T, M, C
            data[..., self.data_dim] = (data[..., self.data_dim] - self.mean) * self.inv_std
        elif data.dim() == 3:  # (N, T, M) or (N, M, T), in case of data dimension C=1
            data = (data - self.mean) * self.inv_std

        return data

    def inverse_transform(self, data):
        if len(data.shape) == 4:  # assume N, T, M, C
            data[..., self.data_dim] = (data[..., self.data_dim] * self.std) + self.mean
        elif len(data.shape) == 3:  # (N, T, M) or (N, M, T), in case of data dimension C=1
            data = (data * self.std) + self.mean

        return data
    
    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.inv_std = self.inv_std.to(device)

    def inverse_transform_plog_sigma(self, plog_sigma):
        
        """ 
            we scale the raw input x to zero mean and unit variance variable u using self.transform:
                u = (x - mean) / std 
            when we scale back, 
                x = u * std + mean 
            the variance should be
                Var(x) = Var(u) * std**2
            and the p-log-sigma should be:
                log Var(x) = log(Var(u)) + 2log(std)
            However, through experiments, we find this biased p-log-sigma are usually too large,
            so we discarded this inductive bias, and let the network learn by itself. 
        """
        return plog_sigma # + 2*torch.log(self.std)
    
    @property
    def device(self):
        return self.mean.device.type
    
    def state_dict(self):
        return {"mean": self.mean.item(), "std": self.std.item(), 
                "data_dim": self.data_dim, "device": self.device}
