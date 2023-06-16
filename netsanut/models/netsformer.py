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
            d_model=hid_dim, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)

    def forward(self, x):
        pass


class NeTSFormer(nn.Module):

    """ Networked Time Series Prediction with Transformer

        The idea resembles the divided space-time attention in TimesFormer for video action recognition, see https://github.com/facebookresearch/TimeSformer
    """

    def __init__(self,
                 in_dim=2,
                 hid_dim=64,
                 ff_dim=512,
                 out_dim=12,
                 nhead=2,
                 dropout=0.1,
                 encoder_layers=2,
                 decoder_layers=2,
                 time_first=True) -> None:

        super().__init__()
        self.feature_embedding = nn.Conv2d(in_dim, hid_dim, kernel_size=(1, 1))
        self.spatial_embed     = LearnedPositionalEncoding(hid_dim, dropout)
        self.temporal_embed    = LearnedPositionalEncoding(hid_dim, dropout)
        self.encoder = nn.ModuleList([DividedTimeSpaceAttentionLayer(
            hid_dim=hid_dim,
            nhead=nhead,
            ff_dim=ff_dim,
            dropout=dropout,
            time_first=time_first
        ) for _ in encoder_layers])
        self.decoder = nn.ModuleList([nn.TransformerDecoderLayer(
            d_model=hid_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        ) for _ in decoder_layers])
        
        self.pred_mean = nn.Linear(in_features=hid_dim, out_features=out_dim)
        self.pred_var = nn.Linear(in_features=hid_dim, out_features=out_dim)

    def forward(self, x):
        x = self.feature_embedding(x)
        x = self.temporal_embed(x)
        x = self.spatial_embed(x)
        x = self.encoder(x)
        x = self.decoder(x)
