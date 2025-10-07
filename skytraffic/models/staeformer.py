""" Implementation modifed from https://github.com/LibCity/Bigscity-LibCity
"""

from logging import getLogger

import torch.nn as nn
import torch
from typing import Dict, Tuple
import numpy as np

from .base import BaseModel
from .layers import masked_mae
from .utils.transform import TensorDataScaler


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
                             query @ key
                     ) / self.head_dim ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
            self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class STAEformer(BaseModel):
    def __init__(
        self,
        # Model-specific parameters with defaults based on original implementation
        steps_per_day: int = 288,
        input_dim: int = 1,
        output_dim: int = 1,
        input_embedding_dim: int = 24,
        tod_embedding_dim: int = 24,
        dow_embedding_dim: int = 24,
        spatial_embedding_dim: int = 0,
        adaptive_embedding_dim: int = 80,
        feed_forward_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_mixed_proj: bool = True,
        add_time_in_day: bool = True,
        add_day_in_week: bool = False,
        loss_ignore_value: float = float("nan"),
        norm_label_for_loss: bool = True,
        # BaseModel parameters
        input_steps: int = 12,
        pred_steps: int = 12,
        num_nodes: int = 207,
        data_null_value: float = 0.0,
        metadata: dict = None,
    ):
        super().__init__(input_steps=input_steps, pred_steps=pred_steps, num_nodes=num_nodes,
                        data_null_value=data_null_value, metadata=metadata)
        
        self._logger = getLogger()

        self.steps_per_day = steps_per_day
        self.input_dim = input_dim + (1 if add_time_in_day else 0) + (1 if add_day_in_week else 0)
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
                input_embedding_dim
                + (tod_embedding_dim if add_time_in_day else 0)
                + (dow_embedding_dim if add_day_in_week else 0)
                + spatial_embedding_dim
                + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj
        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week
        self.loss_ignore_value = loss_ignore_value
        self.norm_label_for_loss = norm_label_for_loss

        # Initialize scaler from metadata if available
        if metadata is not None:
            self.adapt_to_metadata(metadata)

        self.input_proj = nn.Linear(self.input_dim, input_embedding_dim)
        if self.add_time_in_day:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if self.add_day_in_week:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(input_steps, num_nodes, adaptive_embedding_dim))
            )

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                input_steps * self.model_dim, pred_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(input_steps, pred_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
    def make_predictions(self, source):
        """
        Original forward method renamed.
        """
        batch_size = source.shape[0]
        
        x = self.feature_extraction(source)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, input_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.input_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.pred_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, pred_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, input_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, pred_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, pred_steps, num_nodes, output_dim)

        return out.squeeze()
    
    
    def feature_extraction(self, source):
        
        # x: (batch_size, input_steps, num_nodes, input_dim+tod+dow=3)
        x = source
        batch_size = x.shape[0]

        if self.add_time_in_day:
            tod = x[..., 1]
        if self.add_day_in_week:
            dow = x[..., 2]
        x = x[..., : self.input_dim]

        x = self.input_proj(x)  # (batch_size, input_steps, num_nodes, input_embedding_dim)
        features = [x]
        if self.add_time_in_day:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, input_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.add_day_in_week:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, input_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.input_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        x = torch.cat(features, dim=-1)  # (batch_size, input_steps, num_nodes, model_dim)

        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)
        # (batch_size, input_steps, num_nodes, model_dim)

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
