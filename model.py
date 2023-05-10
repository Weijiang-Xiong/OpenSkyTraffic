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


class NetworkedTimeSeriesModel(nn.Module):
    
    def __init__(self, 
                 dropout=0.1, 
                 in_dim=2, 
                 out_dim=12, 
                 rnn_layers=3, 
                 hid_dim=64, 
                 enc_layers=6, 
                 dec_layers=6, 
                 heads=2):
        
        super(NetworkedTimeSeriesModel, self).__init__()
        self.feature_embedding = nn.Conv2d(in_channels=in_dim,
                                           out_channels=hid_dim,
                                           kernel_size=(1, 1))
        self.temp_embedding = TemproalEmbedding(hid_dim, layers=rnn_layers, dropout=dropout)
        self.mean_estimate = nn.Linear(hid_dim, out_dim)
        self.var_estimate = nn.Linear(hid_dim, out_dim)
        self.network = TrafficTransformer(in_dim=hid_dim, enc_layers=enc_layers, dec_layers=dec_layers, dropout=dropout, heads=heads)
        
    def set_fixed_mask(self, adj_mtxs):
        mask = sum([s.detach() for s in adj_mtxs])
        # a True value in self.mask indicates the corresponding key will be ignored
        self.mask = mask == 0
        
    def forward(self, input):
        # input shape: (N, C_in, M, T)
        x = self.feature_embedding(input)  # out shape (N, C_hid, M, T)
        x = self.temp_embedding(x)  # out shape (N, C_hid, M, T)
        x = x[..., -1].transpose(1, 2)  # (N, C_hid, M) => (N, M, C_hid)
        x = self.network(x, self.mask)  # out shape (N, M, C_hid)
        mean = self.mean_estimate(x)  # out shape (N, M, T)
        var = self.var_estimate(x)  # out shape (N, M, T)

        return mean, var  # (N, M, T)

def build_model(args, adjacencies):
    
    model = NetworkedTimeSeriesModel(dropout=args.dropout,
                    in_dim=args.in_dim, 
                    out_dim=args.pred_win, 
                    rnn_layers=args.rnn,
                    hid_dim=args.hid_dim, 
                    enc_layers=args.enc, 
                    dec_layers=args.dec, 
                    heads=args.nhead)
    
    model.set_fixed_mask(adjacencies)
    model.to(torch.device(args.device))
    
    return model 

def test_ttnet_forward():
    
    print("Testing TTNet forward pass...")
    N, C, M, T = 8, 2, 207, 12
    random_input = torch.rand((N, C, M, T))
    random_support = [torch.randint(0, 2, (M, M)).bool() for _ in range(2)]
    model = NetworkedTimeSeriesModel(supports=random_support)
    mean, var = model(random_input)
    assert mean.shape == (N, M, T)
    assert var.shape == (N, M, T)
    print("Test OK")

def test_posenc():
    
    print("Testing positional encoding...")
    
    encoders_batch_first = [
        PositionalEncoding(64, max_len=100, batch_first=True),
        LearnedPositionalEncoding(64, max_len=100, batch_first=True)
        ]
    encoders_not_batch_first = [
        PositionalEncoding(64, max_len=100, batch_first=False),
        LearnedPositionalEncoding(64, max_len=100, batch_first=False)
        ]
    
    for enc in encoders_batch_first:
        print(enc.__class__.__name__)
        out = enc(torch.ones(size=(2, 12, 64)))
        
    for enc in encoders_not_batch_first:
        print(enc.__class__.__name__)
        out = enc(torch.ones(size=(12, 2, 64)))

if __name__ == "__main__":

    test_posenc()
    test_ttnet_forward()