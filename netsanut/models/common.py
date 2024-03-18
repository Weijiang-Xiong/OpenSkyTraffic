""" This file defines commonly used layers
"""

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

class MLP_LazyInput(nn.Module):
    
    def __init__(self, hid_dim, out_dim, dropout, layernorm=True) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.LazyLinear(hid_dim)
        self.norm1 = nn.LayerNorm(hid_dim) if layernorm else nn.Identity() # Identity() basically does nothing 
        self.linear2 = nn.Linear(hid_dim, out_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(self.linear1(x))
        x = self.dropout(torch.relu(x))
        return self.linear2(x)

if __name__ =="__main__":
    pe = LearnedPositionalEncoding(64)
    print(pe)
    pe = PositionalEncoding(64)
    print(pe)