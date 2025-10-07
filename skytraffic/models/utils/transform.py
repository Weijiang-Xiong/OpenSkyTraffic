import torch
from typing import List

class TensorDataScaler:
    """
    normalize the data, a simplified version of sklearn.preprocessing.StandardScaler
    assume the data to have shape
        1. (N, T, P, C) where the 0 in the C dimension is the data, and the rest may be time in day, day in week
        2. (N, T, P) just data, no time appended.

    """

    def __init__(self, mean: float, std: float, data_dim: int = 0):
        """_summary_

        Args:
            mean (float): the mean of the data (should be a scalar or a tensor with 1 element)
            std (float): the standard deviation of the data (should be a scalar or a tensor with 1 element)
            data_dim (int, optional): the index of data sequences of concern (not auxiliary features). Defaults to 0.
        """
        self.data_dim = data_dim
        
        if isinstance(mean, torch.Tensor):
            # this will throw an error if the tensor has more than one element
            # it is better to signal the user that the input can be problematic
            mean = mean.item() 
        if isinstance(std, torch.Tensor):
            std = std.item() 
        assert isinstance(mean, (int, float)), "mean should be a scalar"
        assert isinstance(std, (int, float)), "std should be a scalar"
        
        self.mean = torch.tensor(mean, requires_grad=False)
        self.std = torch.tensor(std, requires_grad=False)
        self.inv_std = 1.0 / self.std

    def transform(self, data, datadim_only: bool = True):
        """ Apply Z-score normalization to the data.
            `datadim_only` is set to True by default, because the major use case is to normalize the input data. 
            Since input data sequences are usually augmented with auxiliary features (like time-in-day encoding), we can only normalize the data dimension.
            The prediction labels are not augmented, so we need to normalize everything in the input tensor. 
        """
        if datadim_only:
            data[..., self.data_dim] = (data[..., self.data_dim] - self.mean) * self.inv_std
        else:
            data = (data - self.mean) * self.inv_std

        return data

    def inverse_transform(self, data, datadim_only: bool = False):
        """ Inverse the Z-score normalization.
            `datadim_only` is set to False by default, because the major use case is to inverse the predictions, which will not contain auxiliary features.
        """
        if datadim_only:
            data[..., self.data_dim] = (data[..., self.data_dim] * self.std) + self.mean
        else:
            data = (data * self.std) + self.mean

        return data
    
    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.inv_std = self.inv_std.to(device)
        return self
    
    @property
    def device(self):
        return self.mean.device.type
    
    def state_dict(self):
        return {"mean": self.mean.item(), "std": self.std.item(), "data_dim": self.data_dim}
