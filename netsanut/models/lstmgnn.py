import torch
import torch.nn as nn
import torch_geometric.nn as gnn

import numpy as np
from scipy.stats import rv_continuous, gennorm

from netsanut.loss import GeneralizedProbRegLoss
from typing import Dict, List, Tuple
from einops import rearrange

from netsanut.data.transform import TensorDataScaler
from .common import MLP_LazyInput, LearnedPositionalEncoding
from .catalog import MODEL_CATALOG

class LSTMGNN(nn.Module):
    def __init__(self, use_global=True, normalize_input=True, scale_output=True, 
                 d_model=64, global_downsample_factor:int=1, layernorm=True, ignore_value: float=-1.0,
                 adjacency_hop: int=1):
        super().__init__()
        
        self.use_global = use_global

        self.normalize_input = normalize_input
        self.scale_output = scale_output
        
        self._calibrated_intervals: Dict[float, float] = dict()
        # self._beta: int = beta
        self._distribution: rv_continuous = None
        # self._set_distribution(beta=1)
        self.ignore_value = ignore_value
        self.adjacency_hop = adjacency_hop
        self.metadata: Dict[str, torch.Tensor] = None

        self.spatial_encoding = LearnedPositionalEncoding(d_model=d_model, max_len=1570)
        
        self.ld_embedding = nn.Conv2d(in_channels=2, out_channels=d_model, kernel_size=(1, 1))
        self.temporal_encoding_ld = LearnedPositionalEncoding(d_model=d_model)
        self.ld_temporal = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=3, batch_first=True)
        self.ld_norm = nn.LayerNorm(d_model) if layernorm else nn.Identity()

        if self.use_global:
            # input dimension can be 64 or 128 depending on the data modalities, so we let it be determined at first forward pass
            global_dim = d_model // global_downsample_factor
            self.channel_down_sample = nn.LazyLinear(out_features=global_dim)
            # embedding, LSTM and MLP already contain dropout, so we only add dropout to the graph convolution
            self.dropout = nn.Dropout(p=0.1) 
            self.relu = nn.ReLU()
            # check the answer from @rusty1s https://github.com/pyg-team/pytorch_geometric/issues/965
            self.gcn_1 = gnn.GCNConv(in_channels=global_dim, out_channels=global_dim, node_dim=1)
            self.gcn_2 = gnn.GCNConv(in_channels=global_dim, out_channels=global_dim, node_dim=1)
            self.gcn_3 = gnn.GCNConv(in_channels=global_dim, out_channels=global_dim, node_dim=1)
            self.global_norm1 = nn.LayerNorm(global_dim) if layernorm else nn.Identity()
            self.global_norm2 = nn.LayerNorm(global_dim) if layernorm else nn.Identity()

            self.channel_up_sample = nn.Linear(in_features=global_dim, out_features=d_model)
            
        self.prediction = MLP_LazyInput(hid_dim=int(d_model * 2), out_dim=12, dropout=0.1)

        self.loss = GeneralizedProbRegLoss(aleatoric=False, exponent=1, ignore_value=self.ignore_value)

    @property
    def device(self):
        return list(self.parameters())[0].device

    @property
    def num_params(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    @property
    def is_probabilistic(self):
        return self._distribution is not None

    def forward(self, data: dict[str, torch.Tensor]):
        """
        time series forecasting task,
        data is assumed to have (N, T, P, C) shape (assumed to be unnormalized)
        label is assumed to have (N, T, P) shape (assumed to be unnormalized)

        compute loss in training mode, predict future values in inference
        """

        # preprocessing (if any)
        source, target = self.preprocess(data)

        if self.training:
            assert target is not None, "label should be provided for training"
            return self.compute_loss(source, target)
        else:
            # we should not have target sequences in inference
            return self.inference(source)

    def preprocess(self, data: dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.normalize_input:
            source = {
                "source": self.datascaler.transform(data["source"].to(self.device)),
            }
        else:
            source = {"source": data["source"].to(self.device)}
        target = {"target": data["target"].to(self.device)}
        
        return source, target

    def post_process(self, prediction: torch.Tensor) -> torch.Tensor:
        if self.scale_output:
            # scale back using the inverse_transform of the output scaler
            prediction["pred"] = self.datascaler.inverse_transform(prediction["pred"])

        return prediction

    def make_prediction(self, source: dict[str, torch.Tensor]) -> torch.Tensor:
        x_ld = source["source"]
        N, _, P, C = x_ld.shape 

        all_mode_features = [] # store the features of all modalities
        
        x_ld = rearrange(x_ld, "N T P C -> N C P T")
        x_ld = self.ld_embedding(x_ld)
        x_ld = rearrange(x_ld, "N C P T -> (N T) P C")
        x_ld = self.spatial_encoding(x_ld)
        x_ld = rearrange(x_ld, "(N T) P C -> (N P) T C", N=N)
        x_ld = self.temporal_encoding_ld(x_ld)
        x_ld, _ = self.ld_temporal(x_ld)
        x_ld = self.ld_norm(x_ld)
        x_ld = rearrange(x_ld[:, -1, :], "(N P) C -> N P C", N=N)
        all_mode_features.append(x_ld)
        
        if self.use_global:
            x_global = self.channel_down_sample(torch.cat(all_mode_features, dim=-1))
            # graph convolution
            x_inter = self.dropout(self.relu(self.global_norm1(self.gcn_1(x_global, self.metadata["edge_index"]))))
            x_inter = self.dropout(self.relu(self.global_norm2(self.gcn_2(x_inter, self.metadata["edge_index"]))))
            x_global = x_global + x_inter
            x_global = self.gcn_3(x_global, self.metadata["edge_index"])
            x_global = self.channel_up_sample(x_global)
            all_mode_features.append(x_global)

        fused_features = torch.cat(all_mode_features, dim=-1)
        
        pred = self.prediction(fused_features)

        return {
            "pred": rearrange(pred, "N P T -> N T P"),
        }

    def compute_loss(self, source: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        pred_res = self.make_prediction(source)
        pred_res = self.post_process(pred_res) # scale back and then compute loss 
        loss = self.loss(pred_res["pred"], target["target"])

        return {"loss": loss}

    def inference(self, source: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.post_process(self.make_prediction(source))

    def adapt_to_metadata(self, metadata):
        
        assert self.training, "metadata should be loaded in training mode"
        
        self.metadata = dict()
        
        self.datascaler = TensorDataScaler(mean=metadata['mean'], std=metadata['std'], data_dim=metadata['data_dim'])
        # adjacency can be one or multiple adjacency matrices 
        if not isinstance(metadata['adjacency'], (list, tuple)):
            metadata['adjacency'] = [metadata['adjacency']]
        adj_mtx = sum([s.detach() for s in metadata['adjacency']])
        binary_adjacency = (adj_mtx > 0)
        
        if isinstance(self.adjacency_hop, int) and self.adjacency_hop > 1:
            # do a loop instead of calling matrix power to avoid numerical problem
            for _ in range(self.adjacency_hop - 1): 
                binary_adjacency = torch.mm(binary_adjacency.float(), binary_adjacency.float())
                binary_adjacency = (binary_adjacency > 0)
                
        edge_index = torch.nonzero(binary_adjacency, as_tuple=False).T
        self.metadata['edge_index'] = edge_index.to(self.device)

    def _set_distribution(self, beta: int) -> rv_continuous:
        if beta is None:
            self._distribution = None
        else:
            self._distribution = gennorm(beta=int(beta))
        return self._distribution
    
    def state_dict(self):
        """ we add datascalar and metadata to the state_dict, so that they will be saved to the checkpoint, 
        and then can be loaded later.
        """
        state = dict()
        state["model_params"] = super().state_dict()
        state["metadata"] = self.metadata
        state["datascaler"] = self.datascaler.state_dict()
        return state
    
    def load_state_dict(self, state_dict):
        self.datascaler = TensorDataScaler(**state_dict["datascaler"])
        self.metadata = state_dict["metadata"]
        super().load_state_dict(state_dict["model_params"])
        
if __name__.endswith("lstmgnn"):
    MODEL_CATALOG.register(LSTMGNN)