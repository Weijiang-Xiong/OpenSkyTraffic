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
from netsanut.util import default_metrics
from netsanut.data import TensorDataScaler
from netsanut.events import get_event_storage

class LearnedPositionalEncoding(nn.Module):
    
    def __init__(self, d_model, dropout=0.1, max_len=500, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        # the input dimension is (batch_size, seq_len, feature_dim) if batch_first is True
        self.batch_dim, self.att_dim = (0, 1) if batch_first else (1, 0)
        self.encoding_dict = nn.Parameter(torch.rand(size=(max_len, d_model)))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        encoding = self.encoding_dict[:x.size(self.att_dim), :].unsqueeze(self.batch_dim)
        x = x + encoding
        return self.dropout(x)


class PositionalEncoding(nn.Module):
    """ modified from Annotated transformer, with batch first flag: 
        https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
    """
    def __init__(self, d_model, dropout=0.1, max_len=500, batch_first=True):
        
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('encoding_dict', pe)
        
        self.batch_first = batch_first
        self.batch_dim, self.att_dim = (0, 1) if batch_first else (1, 0)
        
    def forward(self, x):
        encoding = self.encoding_dict[:x.size(self.att_dim), :].unsqueeze(self.batch_dim)
        x = x + encoding
        return self.dropout(x)


class TemproalEmbedding(nn.Module):
    def __init__(self, in_dim, layers=1, dropout=.1):
        super(TemproalEmbedding, self).__init__()
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


class TTNet(nn.Module):
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
        self.temp_embedding = TemproalEmbedding(hid_dim, layers=rnn_layers, dropout=dropout)
        self.mean_estimate = nn.Linear(hid_dim, out_dim)
        self.var_estimate = nn.Linear(hid_dim, out_dim)
        self.network = TrafficTransformer(in_dim=hid_dim, enc_layers=enc_layers, dec_layers=dec_layers, dropout=dropout, heads=heads)
        self.loss = GeneralizedProbRegLoss(aleatoric=aleatoric, 
                                           exponent=exponent, 
                                           alpha=alpha, 
                                           ignore_value=ignore_value)
        self.datascaler:TensorDataScaler = datascaler if datascaler is not None else TensorDataScaler([50, 0.0], [10, 0.0])
        self.scale_before_loss = True
        self.record_auxiliary_metrics = True
        self.metrics = dict()
        
    @property   
    def device(self):
        return list(self.parameters())[0].device

    def set_fixed_mask(self, adj_mtxs):
        mask = sum([s.detach() for s in adj_mtxs])
        # a True value in self.mask indicates the corresponding key will be ignored
        self.mask = (mask == 0).to(self.device)
        
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
        x = x.permute((0, 3, 2, 1)) # N, T, M, C => N, C, M, T
        x = self.feature_embedding(x)  # out shape (N, C_hid, M, T)
        x = self.temp_embedding(x)  # out shape (N, C_hid, M, T)
        x = x[..., -1].permute((0, 2, 1))  # take output from last time step (N, C_hid, M) => (N, M, C_hid)
        x = self.network(x, self.mask)  # out shape (N, M, C_hid)
        
        mean = self.mean_estimate(x)  # out shape (N, M, T)
        logvar = self.var_estimate(x)  # out shape (N, M, T)
        
        return mean.permute(0, 2, 1), logvar.permute(0, 2, 1)

    
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
        self.set_fixed_mask(metadata['adjacency'])
    
# def build_model(args, adjacencies, datascaler=None):
    
#     model = NTSModel(dropout=args.dropout,
#                     in_dim=args.in_dim, 
#                     out_dim=args.pred_win, 
#                     rnn_layers=args.rnn,
#                     hid_dim=args.hid_dim, 
#                     enc_layers=args.enc, 
#                     dec_layers=args.dec, 
#                     heads=args.num_head,
#                     datascaler=datascaler,
#                     aleatoric=args.aleatoric,
#                     exponent=args.exponent,
#                     alpha=args.alpha)
    
#     model.to(torch.device(args.device))
#     model.set_fixed_mask(adjacencies)
    
#     return model 

