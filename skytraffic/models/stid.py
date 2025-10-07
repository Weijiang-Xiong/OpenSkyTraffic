""" Implementation modifed from https://github.com/LibCity/Bigscity-LibCity
"""

from logging import getLogger

import torch
import torch.nn as nn
from typing import Dict, Tuple
import numpy as np

from .base import BaseModel
from .layers import masked_mae
from .utils.transform import TensorDataScaler


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))  # MLP
        hidden = hidden + input_data  # residual
        return hidden


class STID(BaseModel):
    """
    Paper: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting
    Link: https://arxiv.org/abs/2208.05233
    Official Code: https://github.com/zezhishao/STID
    """

    def __init__(
        self,
        # Model-specific parameters with defaults based on original implementation
        time_intervals: int = 300,  # Time intervals in seconds
        num_block: int = 2,
        time_series_emb_dim: int = 64,
        spatial_emb_dim: int = 8,
        temp_dim_tid: int = 8,  # Time in day dimension
        temp_dim_diw: int = 8,  # Day in week dimension
        if_spatial: bool = True,
        if_time_in_day: bool = True,
        if_day_in_week: bool = False,
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
        super().__init__(input_steps=input_steps, pred_steps=pred_steps, num_nodes=num_nodes,
                        data_null_value=data_null_value, metadata=metadata)
        
        self.time_intervals = time_intervals
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_block = num_block
        self.time_series_emb_dim = time_series_emb_dim
        self.spatial_emb_dim = spatial_emb_dim
        self.temp_dim_tid = temp_dim_tid
        self.temp_dim_diw = temp_dim_diw
        self.if_spatial = if_spatial
        self.if_time_in_day = if_time_in_day
        self.if_day_in_week = if_day_in_week
        self.loss_ignore_value = loss_ignore_value
        self.norm_label_for_loss = norm_label_for_loss

        assert (24 * 60 * 60) % self.time_intervals == 0, "time_of_day_size should be Int"
        self.time_of_day_size = int((24 * 60 * 60) / self.time_intervals)
        self.day_of_week_size = 7

        self._logger = getLogger()

        # Initialize scaler from metadata if available
        if metadata is not None:
            self.adapt_to_metadata(metadata)

        if self.if_spatial:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.spatial_emb_dim))
            nn.init.xavier_uniform_(self.node_emb)
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.output_dim * self.input_steps, out_channels=self.time_series_emb_dim, kernel_size=(1, 1),
            bias=True)

        # encoding
        self.hidden_dim = self.time_series_emb_dim + self.spatial_emb_dim * int(self.if_spatial) + \
                          self.temp_dim_tid * int(self.if_day_in_week) + self.temp_dim_diw * int(self.if_time_in_day)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_block)])

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.pred_steps, kernel_size=(1, 1), bias=True)

    def make_predictions(self, source):
        """
        Original forward method renamed.
        """
        hidden = self.feature_extraction(source)
        prediction = self.regression_layer(hidden)

        return prediction.squeeze()

    def feature_extraction(self, source):
        time_series = source[..., :1]

        if self.if_time_in_day:
            tid_data = source[..., 1]
            time_in_day_emb = self.time_in_day_emb[(tid_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)]
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            diw_data = torch.argmax(source[..., 2:], dim=-1)
            day_in_week_emb = self.day_in_week_emb[(diw_data[:, -1, :]).type(torch.LongTensor)]
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = time_series.shape
        time_series = time_series.transpose(1, 2).contiguous()
        time_series = time_series.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(time_series)

        node_emb = []
        if self.if_spatial:
            node_emb.append(self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))

        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)  # concat all embeddings

        hidden = self.encoder(hidden)

        return hidden

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