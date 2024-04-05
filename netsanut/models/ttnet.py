""" Notation of shapes:
        N: batch size
        T: sequence length (time steps)
        C: channel dimension, feature dimension
        M: number of nodes (sensors)
"""

import torch
import numpy as np
import torch.nn as nn

from netsanut.loss import GeneralizedProbRegLoss
from netsanut.data import TensorDataScaler
from netsanut.utils.events import get_event_storage

from .common import PositionalEncoding, LearnedPositionalEncoding
from .base import BaseModel
from .catalog import MODEL_CATALOG

import warnings
warnings.filterwarnings("ignore")

class TemporalEmbedding(nn.Module):
    def __init__(self, in_dim, layers=1, dropout=.1):
        super(TemporalEmbedding, self).__init__()
        self.rnn = nn.LSTM(input_size=in_dim, hidden_size=in_dim, num_layers=layers, dropout=dropout, batch_first=True)

    def forward(self, input):
        N, C, M, T = input.shape
        x = input.permute(0, 2, 3, 1)  # (N, C, M, T) => (N, M, T, C)
        x = x.reshape(N*M, T, C)  # (N, M, T, C) => (N*M, T, C)
        x, _ = self.rnn(x)
        x = x.reshape(N, M, T, C)  # (N*M, T, C) => (N, M, T, C)
        x = x.permute(0, 3, 1, 2)  # (N, M, T, C) => (N, C, M, T)
        return x


class TrafficTransformer(nn.Module):
    def __init__(self, in_dim, enc_layers=1, dec_layers=1, dropout=.1, heads=4):
        super().__init__()
        self.heads = heads
        self.pos = PositionalEncoding(in_dim, dropout=dropout)
        self.lpos = LearnedPositionalEncoding(in_dim, dropout=dropout)
        self.trans = nn.Transformer(in_dim, heads, num_encoder_layers=enc_layers, num_decoder_layers=dec_layers, dim_feedforward=in_dim*4, dropout=dropout, batch_first=True)

    def forward(self, x, mask):
        x = self.pos(x)
        x = self.lpos(x)
        x = self.trans(x, x, tgt_mask=mask)  # TODO this is probably problematic because the second input should be query (if we don't do autoregressive prediction, like DETR)
        return x

    def _gen_mask(self, input):
        l = input.shape[1]
        mask = torch.eye(l)
        mask = mask.bool()
        return mask

@MODEL_CATALOG.register()
class TTNet(BaseModel):
    """ Traffic Transformer Network modified from, code partly rewritten, fixed a few bugs ... 
        https://github.com/R0oup1iao/Traffic-Transformer
        
        Please note this repo is a different method from "Traffic transformer: Capturing the continuity and
        periodicity of time series for traffic forecasting" <https://onlinelibrary.wiley.com/doi/pdf/10.1111/tgis.12644>
    """
    def __init__(self, 
                 dropout=0.1, 
                 in_dim=2, 
                 out_dim=12, 
                 rnn_layers=3, 
                 hid_dim=64, 
                 enc_layers=6, 
                 dec_layers=6, 
                 heads=2,
                 datascaler=None,
                 aleatoric=False,
                 exponent=1,
                 alpha=1.0,
                 ignore_value=0.0,
                 **kwargs):
        
        super(TTNet, self).__init__()
        self.feature_embedding = nn.Conv2d(in_channels=in_dim,
                                           out_channels=hid_dim,
                                           kernel_size=(1, 1))
        self.temp_embedding = TemporalEmbedding(hid_dim, layers=rnn_layers, dropout=dropout)
        self.mean_estimate = nn.Linear(hid_dim, out_dim)
        self.var_estimate = nn.Linear(hid_dim, out_dim)
        self.network = TrafficTransformer(in_dim=hid_dim, enc_layers=enc_layers, dec_layers=dec_layers, dropout=dropout, heads=heads)
        self.loss = GeneralizedProbRegLoss(aleatoric=aleatoric, 
                                           exponent=exponent, 
                                           alpha=alpha, 
                                           ignore_value=ignore_value)
        self.datascaler:TensorDataScaler = datascaler if datascaler is not None else TensorDataScaler([50, 0.0], [10, 0.0])


    def set_fixed_mask(self, adj_mtxs):
        mask = sum([s.detach() for s in adj_mtxs])
        # a True value in self.mask indicates the corresponding key will be ignored
        self.mask = (mask == 0).to(self.device)


    def make_pred(self, x:torch.Tensor):
        """ 
            run a forward pass through the network 
            data is of shape (N, T, M, C_in), assumed to be normalized 
            output shape (N, T, M)
        """
        x = x.permute((0, 3, 2, 1)) # N, T, M, C => N, C, M, T
        x = self.feature_embedding(x)  # out shape (N, C_hid, M, T)
        x = self.temp_embedding(x)  # out shape (N, C_hid, M, T)
        x = x[..., -1].permute((0, 2, 1))  # take output from last time step (N, C_hid, M) => (N, M, C_hid)
        x = self.network(x, self.mask)  # out shape (N, M, C_hid)
        
        mean = self.mean_estimate(x)  # out shape (N, M, T)
        plog_sigma = self.var_estimate(x)  # out shape (N, M, T)
        
        return mean.permute(0, 2, 1), plog_sigma.permute(0, 2, 1)

    
    def inference(self, source, target=None):
        
        """ if label is None, this is basically the same as self.make_pred 
            one can use this as an alternative to writing torch.no_grad outside the model 
            if label is provided, auxiliary metrics will be computed according to the flag `self.record_auxiliary_metrics`
        """
        
        with torch.no_grad():
            mean, plog_sigma = self.make_pred(source)
            
        mean = self.datascaler.inverse_transform(mean)
        plog_sigma = self.datascaler.inverse_transform_plog_sigma(plog_sigma)
        
        return {"pred": mean, "plog_sigma": plog_sigma}
        
        
    def compute_loss(self, source, target):
        
        mean, plog_sigma = self.make_pred(source)
        
        # scale back the predictions and then calculate loss 
        mean = self.datascaler.inverse_transform(mean)
        plog_sigma = self.datascaler.inverse_transform_plog_sigma(plog_sigma)
        loss = self.loss(mean, target, plog_sigma)
            
        loss_dict = {"loss": loss}
            
        return loss_dict
    
    def adapt_to_metadata(self, metadata):
        
        self.datascaler = TensorDataScaler(mean=metadata['mean'], std=metadata['std'])
        self.set_fixed_mask(metadata['adjacency'])
    