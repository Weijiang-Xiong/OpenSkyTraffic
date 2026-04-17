""" Implementation modifed from https://github.com/LibCity/Bigscity-LibCity
"""

from typing import Dict

import torch
import torch.nn as nn

from .base import BaseModel
from .layers import masked_mae


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


class STIDNet(BaseModel):
    """
    Paper: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting
    Link: https://arxiv.org/abs/2208.05233
    Official Code: https://github.com/zezhishao/STID
    """

    def __init__(
        self,
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
        input_steps: int = 12,
        pred_steps: int = 12,
        num_nodes: int = None,
    ):
        super().__init__()

        self.time_intervals = time_intervals
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.num_nodes = num_nodes
        self.num_block = num_block
        self.time_series_emb_dim = time_series_emb_dim
        self.spatial_emb_dim = spatial_emb_dim
        self.temp_dim_tid = temp_dim_tid
        self.temp_dim_diw = temp_dim_diw
        self.if_spatial = if_spatial
        self.if_time_in_day = if_time_in_day
        self.if_day_in_week = if_day_in_week

        assert (24 * 60 * 60) % self.time_intervals == 0, "time_of_day_size should be Int"
        self.time_of_day_size = int((24 * 60 * 60) / self.time_intervals)
        self.day_of_week_size = 7

        if self.if_spatial:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.spatial_emb_dim))
            nn.init.xavier_uniform_(self.node_emb)
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.output_dim * self.input_steps, out_channels=self.time_series_emb_dim, kernel_size=(1, 1),
            bias=True)

        self.hidden_dim = self.time_series_emb_dim + self.spatial_emb_dim * int(self.if_spatial) + \
                          self.temp_dim_tid * int(self.if_day_in_week) + self.temp_dim_diw * int(self.if_time_in_day)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_block)])

        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.pred_steps, kernel_size=(1, 1), bias=True)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        hidden = self.feature_extraction(source)
        prediction = self.regression_layer(hidden)
        return prediction.squeeze()

    def feature_extraction(self, source: torch.Tensor) -> torch.Tensor:
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

    def compute_loss(self, source: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred = self.forward(source)
        loss_val = masked_mae(pred, target, null_val=float("nan"))
        return {"loss": loss_val}
