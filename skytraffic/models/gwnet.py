""" Implementation modifed from https://github.com/LibCity/Bigscity-LibCity
    1. Deleted adjacency related functions, as this is done in the dataset class
    2. changed Conv1d to Conv2d (behavior changed in torch > 1.11) suggested by https://github.com/nnzhan/Graph-WaveNet/issues/34
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
from typing import Dict, Tuple
import numpy as np

from .base import BaseModel
from .layers import masked_mae
from .utils.transform import TensorDataScaler


class NConv(nn.Module):
    def __init__(self):
        super(NConv, self).__init__()

    def forward(self, x, adj):
        x = torch.einsum('ncvl,vw->ncwl', (x, adj))
        return x.contiguous()


class Linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(Linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(GCN, self).__init__()
        self.nconv = NConv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GWNET(BaseModel):
    def __init__(
        self,
        # Model-specific parameters with defaults based on original implementation
        dropout: float = 0.3,
        blocks: int = 4,
        layers: int = 2,
        gcn_bool: bool = True,
        addaptadj: bool = True,
        # adjtype: str = 'doubletransition',
        randomadj: bool = True,
        aptonly: bool = True,
        kernel_size: int = 2,
        nhid: int = 32,
        residual_channels: int = None,
        dilation_channels: int = None,
        skip_channels: int = None,
        end_channels: int = None,
        apt_layer: bool = True,
        feature_dim: int = 2,
        output_dim: int = 1,
        loss_ignore_value: float = float("nan"),
        norm_label_for_loss: bool = True,
        # BaseModel parameters
        input_steps: int = 12,
        pred_steps: int = 12,
        num_nodes: int = None,
        data_null_value: float = 0.0,
        metadata: dict = None,
    ):
        super().__init__(input_steps=input_steps, pred_steps=pred_steps, num_nodes=num_nodes, data_null_value=data_null_value, metadata=metadata)
        
        # Set up parameters
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        # self.adjtype = adjtype
        self.randomadj = randomadj
        self.aptonly = aptonly
        self.kernel_size = kernel_size
        self.nhid = nhid
        self.residual_channels = residual_channels or self.nhid
        self.dilation_channels = dilation_channels or self.nhid
        self.skip_channels = skip_channels or self.nhid * 8
        self.end_channels = end_channels or self.nhid * 16
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.loss_ignore_value = loss_ignore_value
        self.norm_label_for_loss = norm_label_for_loss
        
        self.apt_layer = apt_layer
        if self.apt_layer:
            self.layers = int(
                np.round(np.log((((self.input_steps - 1) / (self.blocks * (self.kernel_size - 1))) + 1)) / np.log(2)))

        self._logger = getLogger()
        
        # Initialize scaler from metadata if available
        if metadata is not None:
            self.adapt_to_metadata(metadata)

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=self.feature_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))
        
        # the original implementation calculates the adjacency matrix here, but we do it in the dataset class
        # self.cal_adj(self.adjtype) 
        self.supports = [torch.tensor(i).to(self.device) for i in self.adjacency]
        if self.randomadj:
            self.aptinit = None
        else:
            self.aptinit = self.supports[0]
        if self.aptonly:
            self.supports = None

        receptive_field = self.output_dim

        self.supports_len = 0
        if self.supports is not None:
            self.supports_len += len(self.supports)

        if self.gcn_bool and self.addaptadj:
            if self.aptinit is None:
                if self.supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, 10).to(self.device),
                                             requires_grad=True).to(self.device)
                self.nodevec2 = nn.Parameter(torch.randn(10, self.num_nodes).to(self.device),
                                             requires_grad=True).to(self.device)
                self.supports_len += 1
            else:
                if self.supports is None:
                    self.supports = []
                m, p, n = torch.svd(self.aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(self.device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(self.device)
                self.supports_len += 1

        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=self.residual_channels,
                                                   out_channels=self.dilation_channels,
                                                   kernel_size=(1, self.kernel_size), dilation=new_dilation))
                self.gate_convs.append(nn.Conv2d(in_channels=self.residual_channels,
                                                 out_channels=self.dilation_channels,
                                                 kernel_size=(1, self.kernel_size), dilation=new_dilation))
                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=self.dilation_channels,
                                                     out_channels=self.residual_channels,
                                                     kernel_size=(1, 1)))
                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=self.dilation_channels,
                                                 out_channels=self.skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(self.residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(GCN(self.dilation_channels, self.residual_channels,
                                          self.dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                    out_channels=self.end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.pred_steps,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.receptive_field = receptive_field
        self._logger.info('receptive_field: ' + str(self.receptive_field))

    def make_predictions(self, source: torch.Tensor):
        """
        Original forward method renamed.
        """
        x = self.feature_extraction(source)
        x = self.end_conv_2(x)
        # (batch_size, pred_steps, num_nodes, output_dim)
        return x.squeeze()

    def feature_extraction(self, source):
        inputs = source  # (batch_size, input_steps, num_nodes, feature_dim)
        inputs = inputs.transpose(1, 3)  # (batch_size, feature_dim, num_nodes, input_steps)
        inputs = nn.functional.pad(inputs, (1, 0, 0, 0))  # (batch_size, feature_dim, num_nodes, input_steps+1)

        in_len = inputs.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(inputs, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = inputs
        x = self.start_conv(x)  # (batch_size, residual_channels, num_nodes, self.receptive_field)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except(Exception):
                skip = 0
            skip = s + skip
            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        return x

    def preprocess(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        source = data["source"].to(self.device)  # (N, T, P, C)
        target = data["target"].to(self.device) # (N, T, P)
            
        # replace the label values with nan, so that they will be ignored in the loss after normalization
        if np.isnan(self.data_null_value):
            target[target.isnan()] = self.loss_ignore_value
        else:
            target[target == self.data_null_value] = self.loss_ignore_value

        # normalize the data
        source = self.datascaler.transform(source)
        if self.norm_label_for_loss:
            target = self.datascaler.transform(target, datadim_only=False)
        
        return source, target

    def compute_loss(self, source: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        # compute loss at original data scale
        pred = self.make_predictions(source)
        # when label is scaled, we directly train the model to predict the scaled label
        # otherwise, we scale back the prediction and then compute the loss
        if self.norm_label_for_loss:
            loss_val = masked_mae(pred, target, null_val=self.loss_ignore_value)
        else:
            pred = self.datascaler.inverse_transform(pred)
            loss_val = masked_mae(pred, target, null_val=self.data_null_value)
        return {"loss": loss_val}

    def inference(self, source: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred = self.make_predictions(source)
        pred = self.datascaler.inverse_transform(pred)
        return {"pred": pred}

    def adapt_to_metadata(self, metadata):
        self.datascaler = TensorDataScaler(mean=metadata['mean'], std=metadata['std'], data_dim=metadata['data_dim'])

    def to(self, device: torch.device):
        self.datascaler = self.datascaler.to(device)
        return super().to(device)

    def state_dict(self):
        state = dict()
        state["model_params"] = super().state_dict()
        state["datascaler"] = self.datascaler.state_dict()
        return state

    def load_state_dict(self, state_dict, strict: bool = True):
        self.datascaler = TensorDataScaler(**state_dict["datascaler"])
        super().load_state_dict(state_dict["model_params"], strict=strict)