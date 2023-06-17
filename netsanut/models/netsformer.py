""" Notation of shapes:
        N: batch size
        T: sequence length (time steps)
        C: channel dimension, feature dimension
        M: number of nodes (sensors)

    Assumed input shape: (N, T, M, C_feat), where C_feat is the feature dimension

    Output shape: (N, T, M, C_out), where C_out is out dimension, squeezed to (N, T, M) if C_out is 1
"""

import torch
import torch.nn as nn

from typing import List
from einops import rearrange

from netsanut.loss import GeneralizedProbRegLoss
from netsanut.util import default_metrics
from netsanut.data import TensorDataScaler
from .common import LearnedPositionalEncoding, PositionalEncoding


# TODO replace shape manipulations with einops https://einops.rocks/
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

    def forward(self, x, mask=None):
        N, T, M, C = x.shape 
        if self.time_first:
            x = rearrange(x, 'N T M C -> (N M) T C')
            x = self.temporal_attention(x, mask)
            x = rearrange(x, '(N M) T C -> (N T) M C', M=M)
            x = self.space_attention(x, mask)
            x = rearrange(x, '(N T) M C -> N T M C', N=N)
        else:
            x = rearrange(x, 'N T M C -> (N T) M C')
            x = self.space_attention(x, mask)
            x = rearrange(x, '(N T) M C -> (N M) T C', T=T)
            x = self.temporal_attention(x, mask)
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
        
    def forward(self, x, mask=None):
        layer: DividedTimeSpaceAttentionLayer
        for layer in self.layers:
            x = layer(x, mask)
        return x 


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
        
    def forward(self, query, memory, mask):
        layer: nn.TransformerDecoderLayer
        for layer in self.layers:
            x = layer(query, memory, mask)
        return x 

class NeTSFormer(nn.Module):

    """ Networked Time Series Prediction with Transformer

        The idea resembles the divided space-time attention in TimesFormer for video action recognition, see https://github.com/facebookresearch/TimeSformer
    """

    def __init__(self,
                 in_dim: int = 2,
                 hid_dim: int = 64,
                 ff_dim: int = 512,
                 out_dim: int = 12,
                 nhead: int = 2,
                 dropout: int = 0.1,
                 encoder_layers: int = 2,
                 decoder_layers: int = 2,
                 time_first: bool = True,
                 auxiliary_metrics: bool = True, 
                 reduction: str = "mean",  # loss related
                 aleatoric: bool = False,
                 exponent: int = 1,
                 alpha: float = 1.0,
                 ignore_value: float = 0.0) -> None:

        super().__init__()
        
        self.feature_embedding = nn.Conv2d(in_dim, hid_dim, kernel_size=(1, 1))
        self.spatial_encoding     = LearnedPositionalEncoding(hid_dim, dropout)
        self.temporal_encoding    = LearnedPositionalEncoding(hid_dim, dropout)
        
        self.encoder = SpatialTemporalEncoder(
            num_layers=encoder_layers,
            hid_dim=hid_dim,
            nhead=nhead,
            ff_dim=ff_dim,
            dropout=dropout,
            time_first=time_first
        )
        self.temporal_aggregate = lambda x: torch.mean(x, 1) # simply take the mean
        self.decoder = SpatialDecoder(
            num_layers=decoder_layers,
            hid_dim=hid_dim,
            nhead=nhead,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        
        self.pred_mean = nn.Linear(in_features=hid_dim, out_features=out_dim)
        self.pred_var = nn.Linear(in_features=hid_dim, out_features=out_dim)
        
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
        
    @property   
    def device(self):
        return list(self.parameters())[0].device
        
    def forward(self, data, label=None):
        """
            time series forecasting task, 
            data is assumed to have (N, T, M, C) shape (assumed to be unnormalized)
            label is assumed to have (N, T, M) shape (assumed to be unnormalized)
            
            compute loss in training mode, predict future values in inference
        """
        
        # normalize data here
        data = self.datascaler.transform(data)
        
        if self.training:
            assert label is not None, "label should be provided for training"
            return self.compute_loss(data, label)
        else:
            return self.inference(data, label)

    def make_pred(self, x:torch.Tensor):
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

    
    def inference(self, data, label=None):
        
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
        
        self.metrics.update({"logvar": logvar})
        
        return mean
        
        
    def compute_loss(self, data, label):
        
        mean, logvar = self.make_pred(data)
        
        # scale back the predictions and then calculate loss 
        mean = self.datascaler.inverse_transform(mean)
        logvar = self.datascaler.inverse_transform_logvar(logvar)
        loss = self.loss(mean, label, logvar)
            
        loss_dict = {"loss": loss}
        
        if self.record_auxiliary_metrics:
            self.metrics = self.compute_auxiliary_metrics(mean, label)
            
        return loss_dict
    
    
    def compute_auxiliary_metrics(self, pred, label):
        
        # prevent gradient calculation when providing auxiliary metrics in training mode
        with torch.no_grad():
            metrics = default_metrics(pred, label)
            
        return {k:v for k, v in metrics.items()}
    
    
    def pop_auxiliary_metrics(self, scalar_only=True):
        """ auxiliary metrics may also contain the log variance, 
            so use scalar_only to control the behavior
        """
        if not self.record_auxiliary_metrics:
            return {} 
        
        if scalar_only:
            scalar_metrics = {k:v for k, v in self.metrics.items() if isinstance(v, (int, float))}
        else:
            scalar_metrics = self.metrics
            
        self.metrics = dict() 
        return scalar_metrics
    
    def visualize(self, data, label):
        preds = self.inference(data)
        raise NotImplementedError 
    
    def adapt_to_metadata(self, metadata):
        
        self.datascaler = TensorDataScaler(mean=metadata['mean'], std=metadata['std'])
        # adjacency can be one or multiple adjacency matrices 
        if not isinstance(metadata['adjacency'], (list, tuple)):
            metadata['adjacency'] = [metadata['adjacency']]
        mask = sum([s.detach() for s in metadata['adjacency']])
        # a True value in self.mask indicates the corresponding key will be ignored
        self.mask = (mask == 0).to(self.device)