""" This file defines commonly used layers
"""
import numpy as np 

import torch 
import torch.nn as nn 
from einops import repeat

class LearnedPositionalEncoding(nn.Module):
    """ implement learned positional encoding layer

        Assumed input shape: (batch_size, seq_len, feature_dim) if `batch_first` is `True`,
        otherwise it will be (seq_len, batch_size, feature_dim)
    """
    def __init__(self, d_model, dropout=0.1, batch_first=True, max_len=500, init_method='rand'):
        super().__init__()
        self.batch_first = batch_first
        self.init_method = init_method
        # the input dimension is (batch_size, seq_len, feature_dim) if batch_first is True
        self.batch_dim, self.att_dim = (0, 1) if batch_first else (1, 0)
        match init_method:
            case 'zero':
                init_values = torch.zeros(size=(max_len, d_model))
            case _: # elif init_method == 'rand'
                init_values = torch.rand(size=(max_len, d_model))
        self.encoding_dict = nn.Parameter(init_values)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        encoding = self.encoding_dict[:x.size(self.att_dim), :].unsqueeze(self.batch_dim)
        x = x + encoding
        return self.dropout(x)
    
    def encodings(self, x):
        """ return the encodings vectors for the input, shape will be the same as input
        """
        encoding = self.encoding_dict[:x.size(self.att_dim), :].unsqueeze(self.batch_dim)
        # repeat along the batch dimension
        encoding = repeat(encoding, 'N M C -> (N r) M C', r=x.size(0))

        return encoding
    
    def extra_repr(self) -> str:
        return "d_model={}, max_len={}, init_method={}".format(
            self.encoding_dict.size(1), self.encoding_dict.size(0), repr(self.init_method)
        )


class PositionalEncoding(nn.Module):
    """ modified from Annotated transformer, with batch first flag: 
        https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding

        Assumed input shape: (batch_size, seq_len, feature_dim) if `batch_first` is `True`,
        otherwise it will be (seq_len, batch_size, feature_dim)
    """
    def __init__(self, d_model, dropout=0.1, max_len=500, batch_first=True):
        
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor([10000.0])) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('encoding_dict', pe)
        
        self.batch_first = batch_first
        self.batch_dim, self.att_dim = (0, 1) if batch_first else (1, 0)
        
    def forward(self, x):
        encoding = self.encoding_dict[:x.size(self.att_dim), :].unsqueeze(self.batch_dim)
        x = x + encoding
        return self.dropout(x)
    
    def encodings(self, x):
        """ return the encodings vectors for the input, shape will be the same as input
        """
        encoding = self.encoding_dict[:x.size(self.att_dim), :].unsqueeze(self.batch_dim)
        # repeat along the batch dimension
        encoding = repeat(encoding, 'N M C -> (N r) M C', r=x.size(0))

        return encoding
    
    def extra_repr(self) -> str:
        return "d_model={}, max_len={}".format(
            self.encoding_dict.size(1), self.encoding_dict.size(0)
        )

class MLP(nn.Module):
    
    def __init__(self, in_dim, hid_dim, out_dim, dropout, layernorm=True) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.norm1 = nn.LayerNorm(hid_dim) if layernorm else nn.Identity() # Identity() basically does nothing 
        self.linear2 = nn.Linear(hid_dim, out_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(self.linear1(x))
        x = self.dropout(torch.relu(x))
        return self.linear2(x)

class ValueEmbedding(nn.Module):
    """ This layer applies an embedding to spatio-temporal traffic time series data with 
    considerations on two tpes of missing values: a) empty and b) unmonitored. A value is
    empty if we have sensor to monitor a certain location at certain time but observed no 
    vehicles, and unmonitored if we have no sensor at all. 
    
    Both cases are represented with `NaN`, this is to make sure that if the invalid values
    are not handled properly, one is likely to have a NaN in output, which is a clear error.
    We prefer no to replace the `NaN` values with place holders like `-1`, because such values
    may results in slient errors.
    
    The embedding layer will apply a simple linear transformation to the valid values and 
    replace the NaN values with corresponding tokens. 
    """
    def __init__(self, d_model:int, assume_clean_input=False) -> None:
        super().__init__()
        self.d_model = d_model
        self.assume_clean_input = assume_clean_input # do not replace NaN values with learnable tokens if True
        self.time_emb_w = nn.Parameter(torch.randn(1, d_model))
        self.time_emb_b = nn.Parameter(torch.randn(1, d_model))
        self.value_emb_w = nn.Parameter(torch.randn(1, d_model))
        self.value_emb_b = nn.Parameter(torch.randn(1, d_model))
        self.empty_token = nn.Parameter(torch.randn(d_model)) # fit the tensor shape N, C, H, W
        self.unmonitored_token = nn.Parameter(torch.randn(d_model))
        
    def forward(self, x: torch.Tensor, invalid_value = torch.nan, monitor_mask: torch.Tensor = None):
        """
        Args:
            x (torch.Tensor): networked timeseries data with shape (N, T, P, 2)
            invalid_value : Defaults to torch.nan.
            monitor_mask (torch.Tensor, optional): A boolean tensor with shape (N, T,P) whose `True` corresponds to the state of monitored, and `False` means unmonitored. Defaults to None.
        """
        N, T, P, _ = x.shape
        value, time = x[:, :, :, 0], x[:, :, :, 1]
        
        time_emb = time.unsqueeze(-1) * self.time_emb_w + self.time_emb_b
        
        # assume the invalid values are already handled in preprocessing, and don't do it here
        if self.assume_clean_input:
            value_emb = value.unsqueeze(-1) * self.value_emb_w + self.value_emb_b
            return (time_emb + value_emb).contiguous()
        
        # comparing by == doesn't work with nan
        if invalid_value in [float("nan"), np.nan, torch.nan, None]:
            invalid_mask = torch.isnan(value)
        else:
            invalid_mask = (value == invalid_value)
        
        # apply linear embedding to the valid values
        value_emb = torch.empty(size=(N, T, P, self.d_model), device=self.device)
        value_emb[~invalid_mask] = value.unsqueeze(-1)[~invalid_mask] * self.value_emb_w + self.value_emb_b
        
        # replace the invalid values with corresponding tokens
        if monitor_mask is not None:
            # if a location is unmonitored, then the value is replaced with unmonitored token
            value_emb[~monitor_mask] = self.unmonitored_token
            # if a location is monitored but still has no valid value, then it's because we 
            # observe no vehicle. In this case, we replace the value with empty token
            value_emb[invalid_mask & monitor_mask] = self.empty_token
        else:
            value_emb[invalid_mask] = self.empty_token
            
        emb = time_emb + value_emb
        
        return emb.contiguous()

    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def extra_repr(self) -> str:
        return "d_model={}".format(self.d_model)

if __name__ =="__main__":
    pe = LearnedPositionalEncoding(64)
    print(pe)
    pe = PositionalEncoding(64)
    print(pe)