""" Notation of shapes:
        N: batch size
        T: sequence length (time steps)
        C: channel dimension, feature dimension
        M: number of nodes (sensors)
"""

import torch
import numpy as np 
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    def __init__(self,d_model, dropout = 0.1,max_len = 500):
        super().__init__()
        self.learned_pos = nn.Parameter(torch.rand(max_len, 1, d_model))
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        # modified because the previous implementation (which call .data) will lose gradient 
        x = x + self.learned_pos[:x.size(0), ...]
        return self.dropout(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TemproalEmbedding(nn.Module):
    def __init__(self, in_dim, layers=1, dropout = .1):
        super(TemproalEmbedding, self).__init__()
        self.rnn = nn.LSTM(input_size=in_dim,hidden_size=in_dim,num_layers=layers,dropout=dropout)

    def forward(self, input):
        ori_shape = input.shape
        x = input.permute(3, 0, 2, 1) # (N, C, M, T) => (T, N, M, C)
        x = x.reshape(ori_shape[3], ori_shape[0] * ori_shape[2], ori_shape[1]) # (T, N, M, C) => (T, N*M, C)
        x,_ = self.rnn(x)
        x = x.reshape(ori_shape[3], ori_shape[0], ori_shape[2], ori_shape[1]) # (T, N*M, C) => (T, N, M, C)
        x = x.permute(1, 3, 2, 0) # (T, N, M, C) => (N, C, M, T)
        return x

class TrafficTransformer(nn.Module):
    def __init__(self, in_dim, enc_layers=1, dec_layers=1, dropout=.1,heads=4):
        super().__init__()
        self.heads = heads
        self.pos = PositionalEncoding(in_dim,dropout=dropout)
        self.lpos = LearnedPositionalEncoding(in_dim, dropout=dropout)
        self.trans = nn.Transformer(in_dim, heads, num_encoder_layers=enc_layers, num_decoder_layers=dec_layers, dim_feedforward=in_dim*4, dropout=dropout)

    def forward(self,input, mask):
        x = input.permute(1,0,2) # (N, M, C) => (M, N, C), batch_first=False by default 
        x = self.pos(x)
        x = self.lpos(x)
        x = self.trans(x,x,tgt_mask=mask) # TODO this is probably wrong because the second input should be query (if we don't do autoregressive prediction, like DETR)
        return x.permute(1,0,2)

    def _gen_mask(self,input):
        l = input.shape[1]
        mask = torch.eye(l)
        mask = mask.bool()
        return mask

class TTNet(nn.Module):
    def __init__(self, dropout=0.1, supports=None, in_dim=2, out_dim=12, hid_dim=32, enc_layers=6, dec_layers=6, heads=2):
        super(TTNet, self).__init__()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=hid_dim,
                                    kernel_size=(1, 1))
        self.start_embedding = TemproalEmbedding(hid_dim, layers=3, dropout=dropout)
        self.end_conv = nn.Linear(hid_dim, out_dim)
        self.end_conv_var = nn.Linear(hid_dim, out_dim)
        self.network = TrafficTransformer(in_dim=hid_dim, enc_layers=enc_layers, dec_layers=dec_layers, dropout=dropout, heads=heads)

        mask = sum([s.detach() for s in supports])
        # a True value in self.mask indicates the corresponding key will be ignored
        self.mask = mask == 0

    def forward(self, input):
        # input shape: (N, C_in, M, T)
        x = self.start_conv(input) # out shape (N, C_hid, M, T)
        x = self.start_embedding(x)[..., -1] # out shape (N, C_hid, M), take the last step output
        x = x.transpose(1, 2) # (N, C_hid, M) => (N, M, C_hid)
        x = self.network(x, self.mask) # out shape (N, M, C_hid)
        x = self.end_conv(x) # out shape (N, M, T)
        return x.transpose(1,2).unsqueeze(-1) # (N, T, M, 1)
