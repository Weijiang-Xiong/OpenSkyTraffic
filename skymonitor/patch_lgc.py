from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch_geometric.nn as gnn

import numpy as np
from einops import rearrange

from skytraffic.models.utils.transform import TensorDataScaler
from skytraffic.models.layers import MLP, LearnedPositionalEncoding, masked_mae
from skytraffic.models.base import BaseModel

class PatchedMVLSTMGCNConv(BaseModel):
    """ LSTM GCN model with temporal patching for multivariate time series forecasting (traffic speed and flow).
    """
    def __init__(
        self,
        # arguments purely based on model
        use_cvg_mask: bool=True,
        use_global=True,
        feature_dim: int = 3,
        d_model=64,
        temp_patching: int = 3,
        global_downsample_factor: int = 1,
        layernorm=True,
        adjacency_hop: int = 1,
        dropout: float = 0.1,
        loss_ignore_value = float("nan"),
        norm_label_for_loss: bool = True,
        # arguments related to dataset
        input_steps: int = 10,
        pred_steps: int = 10,
        pred_feat : int = 2,
        num_nodes: int = 1570,
        data_null_value: float = 0.0,
        metadata: dict = None,
    ):
        """
        Args:
            use_global: whether to use GNN module for spatial information
            d_model: the dimension of the model
            global_downsample_factor: the factor to downsample the node features for GNN
            layernorm: whether to use layer normalization after GNN layer
            adjacency_hop: the number of hops to compute the adjacency matrix
            invalid_value: the invalid value in the label, will be replaced with NaN in order to be ignored in loss computation
            norm_label_for_loss: whether to compute the loss after the Z-normalization of label
            temporal_patching (int, default to None): down-sample the temporal dimension by this factor using a convolution layer.
            masked_value_embedding: will use the customized ValueEmbedding for the input data if True, otherwise a simple linear transformation.
        """
        super().__init__(input_steps=input_steps, pred_steps=pred_steps, num_nodes=num_nodes, data_null_value=data_null_value, metadata=metadata)
        self.use_global = use_global
        self.loss_ignore_value = loss_ignore_value
        self.adjacency_hop = adjacency_hop
        self.norm_label_for_loss = norm_label_for_loss
        self.temp_patching = temp_patching
        self.edge_index: torch.Tensor
        self.pred_feat = pred_feat
        self.use_cvg_mask = use_cvg_mask

        if metadata is not None:
            self.adapt_to_metadata(metadata)
        
        self.patching = nn.Conv2d(in_channels=feature_dim * temp_patching, out_channels=d_model, kernel_size=(temp_patching, 1), stride=(temp_patching, 1))
        self.spatial_pos_enc = LearnedPositionalEncoding(d_model=d_model, max_len=self.num_nodes)
        self.temporal_pos_enc = LearnedPositionalEncoding(d_model=d_model, max_len=self.input_steps // (temp_patching if temp_patching is not None else 1))
        self.temporal = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=3, batch_first=True)
        self.norm = nn.LayerNorm(d_model) if layernorm else nn.Identity()

        if self.use_global:
            # input dimension can be 64 or 128 depending on the data modalities, so we let it be determined at first forward pass
            global_dim = d_model // global_downsample_factor
            self.channel_down_sample = nn.LazyLinear(out_features=global_dim)
            # embedding, LSTM and MLP already contain dropout, so we only add dropout to the graph convolution
            self.dropout = nn.Dropout(p=dropout) 
            self.relu = nn.ReLU()
            # check the answer from @rusty1s https://github.com/pyg-team/pytorch_geometric/issues/965
            self.gcn_1 = gnn.GCNConv(in_channels=global_dim, out_channels=global_dim, node_dim=1)
            self.gcn_2 = gnn.GCNConv(in_channels=global_dim, out_channels=global_dim, node_dim=1)
            self.gcn_3 = gnn.GCNConv(in_channels=global_dim, out_channels=global_dim, node_dim=1)
            self.global_norm1 = nn.LayerNorm(global_dim) if layernorm else nn.Identity()
            self.global_norm2 = nn.LayerNorm(global_dim) if layernorm else nn.Identity()

            self.channel_up_sample = nn.Linear(in_features=global_dim, out_features=d_model)
            
        self.prediction = MLP(in_dim=d_model * int(1 + self.use_global), hid_dim=int(d_model * 2), out_dim=pred_steps*pred_feat, dropout=dropout)

        self.loss = masked_mae

    def preprocess(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        source = data.get("source", None)
        target = data.get("target", None)
        coverage_mask = data.get("coverage_mask", None)

        if source is None and target is None:
            raise ValueError("Both source and target are None in the input data.")
        
        if source is not None:
            # normalize the data
            source = source.to(self.device)
            source = self.in_datascaler.transform(source, datadim_only=True)
            if self.use_cvg_mask and coverage_mask is not None:
                coverage_mask = coverage_mask.unsqueeze(-1).float().to(self.device)
                source = torch.concatenate([source, coverage_mask], dim=-1)

        if target is not None:
            target = target.to(self.device)
            # replace the label values with nan, so that they will be ignored in the loss after normalization
            if np.isnan(self.data_null_value):
                target[target.isnan()] = self.loss_ignore_value
            else:
                target[target == self.data_null_value] = self.loss_ignore_value
            if self.norm_label_for_loss:
                target = self.out_datascaler.transform(target, datadim_only=False)
        
        return source, target

    def post_process(self, prediction: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        prediction['pred'] = self.out_datascaler.inverse_transform(prediction["pred"])
        return prediction

    def compute_loss(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_res = self.make_prediction(source)
        # when label is normalized, we directly train the model to predict the normalized label
        # otherwise, we scale back the prediction and then compute the loss
        if not self.norm_label_for_loss:
            pred_res = self.post_process(pred_res)
        
        loss = self.loss(pred_res['pred'], target, null_val=self.loss_ignore_value)

        return {"loss": loss}

    def inference(self, source: torch.Tensor) -> torch.Tensor:
        return self.post_process(self.make_prediction(source))

    def make_prediction(self, source: dict[str, torch.Tensor]) -> torch.Tensor:
        fused_features = self.feature_extraction(source)
        pred = self.prediction(fused_features)
        pred = rearrange(pred, "N P (T C) -> N T P C", C=self.pred_feat)

        return {"pred": pred}

    def feature_extraction(self, x: torch.Tensor) -> torch.Tensor:
        N, T, P, C = x.shape 

        all_mode_features = [] # store the features of all modalities
        
        x = rearrange(x, "N (k t) P C -> N (k C) t P", k=self.temp_patching)
        x = self.patching(x)
        x = rearrange(x, "N C t P -> (N t) P C")
        x = self.spatial_pos_enc(x)
        x = rearrange(x, "(N T) P C -> (N P) T C", N=N)
        x = self.temporal_pos_enc(x)
        x, _ = self.temporal(x)
        x = self.norm(x)
        x = rearrange(x[:, -1, :], "(N P) C -> N P C", N=N)
        all_mode_features.append(x)
        
        if self.use_global:
            x_global = self.channel_down_sample(torch.cat(all_mode_features, dim=-1))
            # graph convolution
            x_inter = self.dropout(self.relu(self.global_norm1(self.gcn_1(x_global, self.edge_index))))
            x_inter = self.dropout(self.relu(self.global_norm2(self.gcn_2(x_inter, self.edge_index))))
            x_global = x_global + x_inter
            x_global = self.gcn_3(x_global, self.edge_index)
            x_global = self.channel_up_sample(x_global)
            all_mode_features.append(x_global)

        fused_features = torch.cat(all_mode_features, dim=-1)
        
        return fused_features
    

    def adapt_to_metadata(self, metadata):
        
        self.in_datascaler = TensorDataScaler(
            mean=[stats['mean'] for _, stats in metadata['data_stats']['source'].items()], 
            std=[stats['std'] for _, stats in metadata['data_stats']['source'].items()], 
            data_dim=metadata['data_dim']
        )
        self.out_datascaler = TensorDataScaler(
            mean=[stats['mean'] for _, stats in metadata['data_stats']['target'].items()],
            std=[stats['std'] for _, stats in metadata['data_stats']['target'].items()],
            data_dim=metadata['data_dim']
        )
        # adjacency can be one or multiple adjacency matrices 
        adjacency = metadata['adjacency']
        if not isinstance(adjacency, (list, tuple)):
            adjacency = [adjacency]
        adjacency = [torch.as_tensor(s) if not isinstance(s, torch.Tensor) else s for s in adjacency]
        adj_mtx = sum([s.detach() for s in adjacency])
        binary_adjacency = (adj_mtx > 0)
        
        if isinstance(self.adjacency_hop, int) and self.adjacency_hop > 1:
            # do a loop instead of calling matrix power to avoid numerical problem
            for _ in range(self.adjacency_hop - 1): 
                binary_adjacency = torch.mm(binary_adjacency.float(), binary_adjacency.float())
                binary_adjacency = (binary_adjacency > 0)
                
        edge_index = torch.nonzero(binary_adjacency, as_tuple=False).T
        self.edge_index = edge_index

    def to(self, device: torch.device):
        self.in_datascaler = self.in_datascaler.to(device)
        self.out_datascaler = self.out_datascaler.to(device)
        self.edge_index = self.edge_index.to(device)
        return super().to(device)
    
    def state_dict(self):
        """ we add datascalar and metadata to the state_dict, so that they will be saved to the checkpoint, 
        and then can be loaded later.
        """
        state = dict()
        state["model_params"] = super().state_dict()
        state["edge_index"] = self.edge_index
        state["in_datascaler"] = self.in_datascaler.state_dict()
        state["out_datascaler"] = self.out_datascaler.state_dict()
        return state
    
    def load_state_dict(self, state_dict, strict: bool = False):
        self.in_datascaler = TensorDataScaler(**state_dict["in_datascaler"]).to(self.device)
        self.out_datascaler = TensorDataScaler(**state_dict["out_datascaler"]).to(self.device)
        self.edge_index = state_dict["edge_index"].to(self.device)
        super().load_state_dict(state_dict["model_params"], strict=strict)
