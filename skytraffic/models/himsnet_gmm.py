import logging 

import torch
import torch.nn as nn
import torch_geometric.nn as gnn

import numpy as np

from typing import Dict, List, Tuple
from einops import rearrange

from .utils.transform import TensorDataScaler

from .layers import (
    LearnedPositionalEncoding, 
    ValueEmbedding, 
    MultiHeadAttention, 
    GMMPredictionHead
)

logger = logging.getLogger("default")

class HiMSNet_GMM(nn.Module):
    """basically copied from HiMSNet, and only change the output part."""

    def __init__(
        self,
        use_drone=True,  # encoder settings
        use_ld=True,
        use_global=True,
        normalize_input=True,
        scale_output=True,
        d_model=64,
        dropout: float = 0.1,
        global_downsample_factor: int = 1,
        layernorm=True,
        adjacency_hop=3,
        reg_loss_weight: float = 1.0,
        simple_fillna=False,  # replace NaN values with mean at the begeinning
        rescale_anchors: bool = False,
        zero_init=True,
        map_estimation=False,
        metadata=None,
    ):
        super().__init__()
        self.simple_fillna = simple_fillna
        self.adjacency_hop = adjacency_hop
        self.use_drone = use_drone
        self.use_ld = use_ld
        self.use_global = use_global
        if not (self.use_drone or self.use_ld):
            self.use_drone=True
            logger.warning("Must use at least one data modality, use drone data by default")
        self.normalize_input = normalize_input
        self.scale_output = scale_output
        
        self.ignore_value = -1.0
        self.metadata: Dict[str, torch.Tensor] = None
        self.data_scalers: Dict[str, TensorDataScaler] = None

        self.spatial_encoding = LearnedPositionalEncoding(d_model=d_model, max_len=1570, dropout=dropout)
        
        if self.use_drone:
            self.drone_embedding = ValueEmbedding(d_model=d_model, assume_clean_input=self.simple_fillna)
            self.temporal_encoding_drone = LearnedPositionalEncoding(d_model=d_model, dropout=dropout)
            # drone data has higher temporal resolution, so we use two conv layers to down sample
            self.drone_t_patching_1 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=3)
            self.drone_t_patching_2 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=3)
            self.drone_temporal = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=3, batch_first=True)
            self.drone_norm = nn.LayerNorm(d_model) if layernorm else nn.Identity()
            
        if self.use_ld:
            self.ld_embedding = ValueEmbedding(d_model=d_model, assume_clean_input=self.simple_fillna)
            self.temporal_encoding_ld = LearnedPositionalEncoding(d_model=d_model, dropout=dropout)
            self.ld_temporal = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=3, batch_first=True)
            self.ld_norm = nn.LayerNorm(d_model) if layernorm else nn.Identity()

        if self.use_global:
            # input dimension can be 64 or 128 depending on the data modalities, so we let it be determined at first forward pass
            global_dim = d_model // global_downsample_factor
            self.msg_enc = nn.LazyLinear(out_features=global_dim)
            # embedding, LSTM and MLP already contain dropout, so we only add dropout to the graph convolution
            self.dropout_glb = nn.Dropout(p=dropout) 
            self.relu = nn.ReLU()
            # check the answer from @rusty1s https://github.com/pyg-team/pytorch_geometric/issues/965
            self.gcn_1 = gnn.GCNConv(in_channels=global_dim, out_channels=global_dim, node_dim=1)
            self.gcn_2 = gnn.GCNConv(in_channels=global_dim, out_channels=global_dim, node_dim=1)
            self.gcn_3 = gnn.GCNConv(in_channels=global_dim, out_channels=global_dim, node_dim=1)
            self.global_norm1 = nn.LayerNorm(global_dim) if layernorm else nn.Identity()
            self.global_norm2 = nn.LayerNorm(global_dim) if layernorm else nn.Identity()

            self.msg_dec = nn.Linear(in_features=global_dim, out_features=d_model)

        self.s_feat = self.use_drone + self.use_ld + self.use_global
        self.query_regional = nn.Parameter(torch.randn(4, self.s_feat * d_model))
        self.feature_aggregator = MultiHeadAttention(self.s_feat, self.s_feat * d_model, d_model, d_model, dropout)
        
        # so for simbarca, the segment-level speed is limited to 50 kph, but the 
        # unit of the dataset is m/s, so that's roughly 14 m/s, we divide into 5 intervals
        self.segment_head = GMMPredictionHead(
            in_dim=d_model * self.s_feat, # we have drone, ld and global features
            hid_dim=d_model * 2,
            anchors=[1.4, 4.2, 7.0, 9.8, 12.6], 
            sizes=[1.4, 1.4, 1.4, 1.4, 1.4],
            dropout=dropout,
            loss_ignore_value=self.ignore_value,
            zero_init=zero_init,
            mcd_estimation=map_estimation
        )
        # regional speed can not be that fast (max ~9 m/s), so we use a smaller range for anchors 
        self.regional_head = GMMPredictionHead(
            in_dim=d_model * self.s_feat,
            hid_dim=d_model * 2,
            anchors=[1.5, 4.5, 7.5],
            sizes=[1.5, 1.5, 1.5],
            dropout=dropout,
            loss_ignore_value=self.ignore_value,
            zero_init=zero_init,
            mcd_estimation=map_estimation
        )
        # whether to rescale the anchors based on metadata (will be used in adapt_to_metadata)
        self.rescale_anchors = rescale_anchors
        # weight for the regional task
        self.reg_loss_weight = reg_loss_weight
        self.adapt_to_metadata(metadata)

    @property
    def device(self):
        return list(self.parameters())[0].device

    @property
    def num_params(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    def forward(self, data: dict[str, torch.Tensor]):
        """
        time series forecasting task,
        data is assumed to have (N, T, P, C) shape (assumed to be unnormalized)
        label is assumed to have (N, T, P) shape (assumed to be unnormalized)

        compute loss in training mode, predict future values in inference
        """

        # preprocessing (if any)
        source, target = self.preprocess(data)
        
        segment_feat, regional_feat = self.feature_extraction(source)
        segment_head_out = self.segment_head(segment_feat)
        regional_head_out = self.regional_head(regional_feat)
        
        if self.training:
            assert target is not None, "label should be provided for training"

            segment_loss = self.segment_head.losses(segment_head_out, target["pred_speed"])
            regional_loss = self.regional_head.losses(regional_head_out, target["pred_speed_regional"])
            
            loss_dict = dict()
            loss_dict.update({"segment_{}".format(k): v for k, v in segment_loss.items()})
            loss_dict.update({"regional_{}".format(k): self.reg_loss_weight * v for k, v in regional_loss.items()})
            
            return loss_dict
        
        else:
            # we should not use target sequences in inference
            segment_pred = self.segment_head.inference(segment_head_out)
            regional_pred = self.regional_head.inference(regional_head_out)
            
            pred_speed, seg_mixing, seg_means, seg_log_var = segment_pred
            pred_speed_regional, reg_mixing, reg_means, reg_log_var = regional_pred
            
            return {"pred_speed": pred_speed, 
                    "seg_mixing": seg_mixing,
                    "seg_means": seg_means,
                    "seg_log_var": seg_log_var,
                    "pred_speed_regional": pred_speed_regional,
                    "reg_mixing": reg_mixing,
                    "reg_means": reg_means,
                    "reg_log_var": reg_log_var}

    def preprocess(self, data: dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.normalize_input:
            source = {
                "drone_speed": self.data_scalers["drone_speed"].transform(data["drone_speed"].to(self.device)),
                "ld_speed": self.data_scalers["ld_speed"].transform(data["ld_speed"].to(self.device)),
            }
        else:
            source = {
                "drone_speed": data["drone_speed"].to(self.device),
                "ld_speed": data["ld_speed"].to(self.device),
            }
        
        # load the mask data if available
        try:
            source["ld_mask"] = data["ld_mask"].to(self.device)
            source['drone_mask'] = data['drone_mask'].to(self.device)
        except KeyError:
            pass
        
        # use these to replace NaN values with the mean of the data or other constants
        # and this will simply make the value embedding ineffective
        if self.simple_fillna:
            source["drone_speed"] = source["drone_speed"].nan_to_num(nan=self.data_scalers['drone_speed'].mean)
            source["ld_speed"] = source["ld_speed"].nan_to_num(nan=self.data_scalers['ld_speed'].mean)

        target = {
            "pred_speed": data["pred_speed"].nan_to_num(nan=self.ignore_value).to(self.device),
            "pred_speed_regional": data["pred_speed_regional"].nan_to_num(nan=self.ignore_value).to(self.device),
        }
        
        return source, target

    def feature_extraction(self, source: dict[str, torch.Tensor]) -> torch.Tensor:
        x_drone, x_ld = source["drone_speed"], source["ld_speed"]
        drone_mask, ld_mask = source.get("drone_mask", None), source.get("ld_mask", None)
        N, T_drone, P, C = source["drone_speed"].shape
        T_ld = source["ld_speed"].shape[1]

        all_mode_features = [] # store the features of all modalities
        
        if self.use_drone:
            x_drone = self.drone_embedding(x_drone, monitor_mask = drone_mask)
            x_drone = rearrange(x_drone, "N T P C -> (N P) C T")
            x_drone = self.drone_t_patching_1(x_drone)
            x_drone = self.drone_t_patching_2(x_drone)
            x_drone = rearrange(x_drone, "(N P) C T -> (N T) P C", N=N)
            x_drone = self.spatial_encoding(x_drone)
            x_drone = rearrange(x_drone, "(N T) P C -> (N P) T C", N=N)
            x_drone = self.temporal_encoding_drone(x_drone)
            x_drone, _ = self.drone_temporal(x_drone)  
            x_drone = self.drone_norm(x_drone)
            # we take the last step output from the temporal embeddings (LSTM)
            x_drone = rearrange(x_drone[:, -1, :], "(N P) C -> N P C", N=N)
            all_mode_features.append(x_drone)
        
        if self.use_ld:
            ld_mask = ld_mask.unsqueeze(1).tile(1, T_ld, 1) if ld_mask is not None else ld_mask
            x_ld = self.ld_embedding(x_ld, monitor_mask = ld_mask)
            x_ld = rearrange(x_ld, "N T P C -> (N T) P C")
            x_ld = self.spatial_encoding(x_ld)
            x_ld = rearrange(x_ld, "(N T) P C -> (N P) T C", N=N)
            x_ld = self.temporal_encoding_ld(x_ld)
            x_ld, _ = self.ld_temporal(x_ld)
            x_ld = self.ld_norm(x_ld)
            x_ld = rearrange(x_ld[:, -1, :], "(N P) C -> N P C", N=N)
            all_mode_features.append(x_ld)
        
        if self.use_global:
            x_global = self.msg_enc(torch.cat(all_mode_features, dim=-1))
            # graph convolution
            x_inter = self.dropout_glb(self.relu(self.global_norm1(self.gcn_1(x_global, self.edge_index))))
            x_inter = self.dropout_glb(self.relu(self.global_norm2(self.gcn_2(x_inter, self.edge_index))))
            x_global = x_global + x_inter
            x_global = self.gcn_3(x_global, self.edge_index)
            x_global = self.msg_dec(x_global)
            all_mode_features.append(x_global)

        # segment-level and regional features 
        seg_feat = torch.cat(all_mode_features, dim=-1)
        reg_feat, attn_map = self.feature_aggregator(self.query_regional.unsqueeze(0).tile(N, 1, 1), 
                            seg_feat, seg_feat, mask=None)
        
        return seg_feat, reg_feat

    def adapt_to_metadata(self, metadata):
        
        assert self.training, "metadata should be loaded in training mode"
        
        # keep the tensors and arrays on the same device as the model
        self.data_scalers = {
            name: TensorDataScaler(
                mean=metadata["mean_and_std"][name][0], 
                std=metadata["mean_and_std"][name][1],
                data_dim=0,
            ) for name in metadata["input_seqs"] + metadata["output_seqs"]
        }
        self.cluster_id = torch.as_tensor(metadata["cluster_id"])
        # use K-hop adjacency matrix for graph convolution
        if isinstance(self.adjacency_hop, int) and self.adjacency_hop > 1:
            adj_iter = metadata['adjacency']
            adj_init = adj_iter.detach().clone()
            # do a loop instead of calling matrix power to avoid numerical problem
            for _ in range(self.adjacency_hop - 1):
                adj_iter = torch.mm(adj_iter.float(), adj_init.float())
                adj_iter = (adj_iter > 0)
            self.edge_index = torch.nonzero(adj_iter, as_tuple=False).T
            logger.info("Number of edges in the graph: {}".format(self.edge_index.shape))
    
    def state_dict(self):
        state = dict()
        state["model_params"] = super().state_dict()
        state["cluster_id"] = self.cluster_id
        state["edge_index"] = self.edge_index
        state["data_scalers"] = {name: scaler.state_dict() for name, scaler in self.data_scalers.items()}
        return state

    def load_state_dict(self, state_dict):
        self.cluster_id = state_dict["cluster_id"]
        self.edge_index = state_dict["edge_index"]
        self.data_scalers = {
            name: TensorDataScaler(**state).to(self.device)
            for name, state in state_dict["data_scalers"].items()
        }
        super().load_state_dict(state_dict["model_params"])

    def to(self, device: torch.device | str):
        if isinstance(device, str):
            device = torch.device(device)
        self.data_scalers = {name: scaler.to(device) for name, scaler in self.data_scalers.items()}
        self.cluster_id = self.cluster_id.to(device)
        self.edge_index = self.edge_index.to(device)
        return super().to(device)

