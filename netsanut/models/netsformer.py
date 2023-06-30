""" Notation of shapes:
        N: batch size
        T: sequence length (time steps)
        C: channel dimension, feature dimension
        M: number of nodes (sensors)

    Assumed input shape: (N, T, M, C_feat), where C_feat is the feature dimension

    Output shape: (N, T, M, C_out), where C_out is out dimension, squeezed to (N, T, M) if C_out is 1
"""
import copy
import logging

import torch
import torch.nn as nn

from typing import List, Tuple, Dict, Callable
from einops import rearrange

from netsanut.loss import GeneralizedProbRegLoss
from netsanut.util import default_metrics
from netsanut.data import TensorDataScaler
from .common import LearnedPositionalEncoding, PositionalEncoding
from .base import BaseModel

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger("default")

class DividedTimeSpaceAttentionLayer(nn.Module):

    """
        Compute attention separately on the space dimension and time dimension, set `time_first=True` to compute first for time dimension, otherwise space dimension goes first.

        Assumed input and output shape: (N, C, T, M), so it can be stacked without shape manipulation in between.
    """

    def __init__(self, hid_dim, nhead, ff_dim, dropout, time_first=True) -> None:
        super().__init__()
        self.time_first = time_first
        self.space_attention = nn.TransformerEncoderLayer(
            d_model=hid_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.temporal_attention = nn.TransformerEncoderLayer(
            d_model=hid_dim, 
            nhead=nhead, 
            dim_feedforward=ff_dim, 
            dropout=dropout, 
            batch_first=True
        )

    def forward(self, x:torch.Tensor, s_mask=None, t_mask=None) -> torch.Tensor:
        N, T, M, C = x.shape 
        if self.time_first:
            x = rearrange(x, 'N T M C -> (N M) T C')
            x = self.temporal_attention(x, t_mask)
            x = rearrange(x, '(N M) T C -> (N T) M C', M=M)
            x = self.space_attention(x, s_mask)
            x = rearrange(x, '(N T) M C -> N T M C', N=N)
        else:
            x = rearrange(x, 'N T M C -> (N T) M C')
            x = self.space_attention(x, s_mask)
            x = rearrange(x, '(N T) M C -> (N M) T C', T=T)
            x = self.temporal_attention(x, t_mask)
            x = rearrange(x, '(N M) T C -> N T M C', N=N)
        return x 

class SpatialTemporalEncoder(nn.Module):
    
    def __init__(self, num_layers, hid_dim, nhead, ff_dim, dropout, time_first=True) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DividedTimeSpaceAttentionLayer(
                hid_dim,
                nhead,
                ff_dim,
                dropout,
                time_first
            ) for _ in range(num_layers)])
        
    def forward(self, x, s_mask=None, t_mask=None) -> torch.Tensor:
        layer: DividedTimeSpaceAttentionLayer
        for layer in self.layers:
            x = layer(x, s_mask, t_mask)
        return x 

class MLP(nn.Module):
    
    def __init__(self, in_dim, hid_dim, out_dim) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, out_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.linear2(self.linear1(x))

class TemporalAggregate(nn.Module):
    
    def __init__(self, in_dim:int, out_dim:int=1, mode='linear') -> None:
        super().__init__()
        self.in_dim = in_dim
        self.mode = mode
        assert mode in ['linear', 'last', 'avg']
        match mode:
            case 'linear':
                self.agg = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=(1,1))
            case 'last':
                self.agg = lambda x: x[:, -1, ...]
            case 'avg':
                self.agg = lambda x: torch.mean(x, 1)
                
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        
        # 'N T M C -> N M C'
        x = self.agg(x)
        
        # only squeeze the time dimension, as N could be 1 in the last test batch
        return x.squeeze(1)
    
    def extra_repr(self) -> str:
        return "in_dim={}, mode={}".format(self.in_dim, self.mode)
        
class SpatialDecoder(nn.Module):
    """ stack a few transformer decoder layers, implementation is different from nn.TransformerDecoder
    """
    def __init__(self, num_layers, hid_dim, nhead, ff_dim, dropout) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(
            d_model=hid_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        ) for _ in range(num_layers)])
        
    def forward(self, query, memory, mask) -> torch.Tensor:
        layer: nn.TransformerDecoderLayer
        x = query
        for layer in self.layers:
            x = layer(x, memory, mask)
        return x 

def get_trivial_forward() -> Callable:
    
    def trivial_forward(x: torch.Tensor):
        return x
    
    return copy.deepcopy(trivial_forward)

class NeTSFormer(BaseModel):

    """ Networked Time Series Prediction with Transformer

        The idea resembles the divided space-time attention in TimesFormer for video action recognition, see https://github.com/facebookresearch/TimeSformer
    """

    def __init__(self,
                 in_dim: int = 2,
                 hid_dim: int = 64,
                 ff_dim: int = 256,
                 hist_len: int = 12,
                 pred_len: int = 12,
                 nhead: int = 2,
                 dropout: int = 0.1,
                 encoder_layers: int = 2,
                 decoder_layers: int = 2,
                 time_first: bool = True,
                 temp_aggregate: str = "avg",
                 auxiliary_metrics: bool = True, 
                 reduction: str = "mean",  # loss related
                 aleatoric: bool = False,
                 exponent: int = 1,
                 alpha: float = 1.0,
                 ignore_value: float = 0.0,
                 se_type: str = "learned", # encoding related
                 se_init: str = "rand",
                 te_type: str = "fixed",
                 te_init: str = "",
                 **kwargs) -> None:

        super().__init__()
        
        self.sto_layers = list()
        
        self.feature_embedding = nn.Conv2d(in_dim, hid_dim, kernel_size=(1, 1))
        self.spatial_encoding     = self.build_encoding_layer(hid_dim, dropout, se_type, se_init)
        self.temporal_encoding    = self.build_encoding_layer(hid_dim, dropout, te_type, te_init)
        
        self.encoder = SpatialTemporalEncoder(
            num_layers=encoder_layers,
            hid_dim=hid_dim,
            nhead=nhead,
            ff_dim=ff_dim,
            dropout=dropout,
            time_first=time_first
        )
        self.temporal_aggregate = TemporalAggregate(in_dim=hist_len, mode=temp_aggregate) # reduce time dimension
        self.decoder = SpatialDecoder(
            num_layers=decoder_layers,
            hid_dim=hid_dim,
            nhead=nhead,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        
        self.pred_mean = nn.Linear(in_features=hid_dim, out_features=pred_len)
        self.pred_var = nn.Linear(in_features=hid_dim, out_features=pred_len)
        self.sto_layers.append("pred_var")
        
        self.datascaler: TensorDataScaler
        self.loss = GeneralizedProbRegLoss(
            reduction=reduction,
            aleatoric=aleatoric,
            exponent=exponent,
            alpha=alpha,
            ignore_value=ignore_value
        )
        self.record_auxiliary_metrics = auxiliary_metrics
        self.metrics = dict()

    def make_pred(self, x:torch.Tensor) -> Tuple[torch.Tensor]:
        """ 
            run a forward pass through the network 
            data is of shape (N, T, M, C_in), assumed to be normalized 
            output shape (N, T, M)
        """
        N, T, M, C_in = x.shape
        x = rearrange(x, 'N T M C -> N C M T')
        x = self.feature_embedding(x)
        x = rearrange(x, 'N C M T -> (N M) T C')
        x = self.temporal_encoding(x)
        x = rearrange(x, '(N M) T C -> (N T) M C', M=M)
        x = self.spatial_encoding(x)
        x = rearrange(x, '(N T) M C -> N T M C', N=N)
        x = self.encoder(x)
        x = self.temporal_aggregate(x) # 'N T M C -> N M C'
        x = self.decoder(x, x, self.mask) # 'N M C -> N M C'
        mean = self.pred_mean(x) # 'N M C -> N M T'
        mean = rearrange(mean, 'N M T -> N T M')
        logvar = self.pred_var(x)
        logvar = rearrange(logvar, 'N M T -> N T M')
        
        return mean, logvar

    
    def inference(self, data, label=None) -> torch.Tensor:
        
        """ if label is None, this is basically the same as self.make_pred 
            one can use this as an alternative to writing torch.no_grad outside the model 
            if label is provided, auxiliary metrics will be computed according to the flag `self.record_auxiliary_metrics`
        """
        
        with torch.no_grad():
            mean, logvar = self.make_pred(data)
            
        mean = self.datascaler.inverse_transform(mean)
        logvar = self.datascaler.inverse_transform_logvar(logvar)
        
        if self.record_auxiliary_metrics and label is not None:
            self.metrics = self.compute_auxiliary_metrics(mean, label)
        
        return {"pred": mean, "logvar": logvar}
        
        
    def compute_loss(self, data, label) -> Dict[str, torch.Tensor]:
        
        mean, logvar = self.make_pred(data)
        
        # scale back the predictions and then calculate loss 
        mean = self.datascaler.inverse_transform(mean)
        logvar = self.datascaler.inverse_transform_logvar(logvar)
        loss = self.loss(mean, label, logvar)
            
        loss_dict = {"loss": loss}
        
        if self.record_auxiliary_metrics:
            self.metrics = self.compute_auxiliary_metrics(mean, label)
            
        return loss_dict
    
    
    def adapt_to_metadata(self, metadata) -> None:
        
        self.datascaler = TensorDataScaler(mean=metadata['mean'], std=metadata['std'])
        # adjacency can be one or multiple adjacency matrices 
        if not isinstance(metadata['adjacency'], (list, tuple)):
            metadata['adjacency'] = [metadata['adjacency']]
        mask = sum([s.detach() for s in metadata['adjacency']])
        # a True value in self.mask indicates the corresponding key will be ignored
        self.mask = (mask == 0).to(self.device)
    
    @staticmethod 
    def build_encoding_layer(hid_dim, dropout, encoding_type:str, init_method="rand") -> nn.Module | Callable:
        match encoding_type:
            case "learned":
                return LearnedPositionalEncoding(hid_dim, dropout, init_method=init_method)
            case "fixed":
                return PositionalEncoding(hid_dim, dropout)
            case "None" | "none" | "" | None:
                return get_trivial_forward()
            
    def get_param_groups(self) -> Dict[str, List[nn.Parameter]]:
        
        det_params, sto_params = [], []
        
        for m_name, module in self.named_children():
            for p_name, param in module.named_parameters(remove_duplicate=True, recurse=True):
                if not param.requires_grad:
                    continue
            
                if m_name in self.sto_layers:
                    sto_params.append(("{}.{}".format(m_name, p_name), param))
                else:
                    det_params.append(("{}.{}".format(m_name, p_name), param))
        
        
        return {"det": [param for _, param in det_params],
                "sto": [param for _, param in sto_params]}
        
    def adapt_to_new_config(self, config) -> None:
        
        self.loss = GeneralizedProbRegLoss(**config.loss)
        logger.info("Using new loss function:")
        logger.info("{}".format(self.loss))