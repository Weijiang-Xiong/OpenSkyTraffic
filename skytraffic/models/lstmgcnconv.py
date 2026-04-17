import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from .layers import masked_mae
from typing import Dict
from einops import rearrange

from .layers import MLP, LearnedPositionalEncoding, ValueEmbedding

from .base import BaseModel

class LSTMGCNConv(BaseModel):
    def __init__(
        self,
        # arguments purely based on model
        use_global=True,
        d_model=64,
        global_downsample_factor: int = 1,
        layernorm=True,
        assume_clean_input: bool = True,
        adjacency_hop: int = 1,
        dropout: float = 0.1,
        # arguments related to dataset
        input_steps: int = 12,
        pred_steps: int = 12,
        num_nodes: int = None,
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
        """
        super().__init__()
        self.input_steps = input_steps if input_steps is not None else metadata["input_steps"]
        self.pred_steps = pred_steps if pred_steps is not None else metadata["pred_steps"]
        self.num_nodes = num_nodes if num_nodes is not None else metadata["num_nodes"]
        self.use_global = use_global
        self.assume_clean_input = assume_clean_input
        self.adjacency_hop = adjacency_hop
        self.edge_index: torch.Tensor | None = None
        if metadata is not None:
            self.adapt_to_metadata(metadata)
        
        self.spatial_encoding = LearnedPositionalEncoding(d_model=d_model, max_len=self.num_nodes)
        self.ld_embedding = ValueEmbedding(d_model=d_model, assume_clean_input=self.assume_clean_input)
        self.temporal_encoding_ld = LearnedPositionalEncoding(d_model=d_model, max_len=self.input_steps)
        self.ld_temporal = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=3, batch_first=True)
        self.ld_norm = nn.LayerNorm(d_model) if layernorm else nn.Identity()

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
            
        self.prediction = MLP(in_dim=d_model * int(1 + self.use_global), hid_dim=int(d_model * 2), out_dim=pred_steps, dropout=dropout)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        fused_features = self.feature_extraction(source)
        pred = self.prediction(fused_features)
        return rearrange(pred, "N P T -> N T P")

    def compute_loss(self, source: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred = self(source)
        loss = masked_mae(pred, target, null_val=float("nan"))
        return {"loss": loss}

    def feature_extraction(self, x: torch.Tensor) -> torch.Tensor:
        N, _, P, C = x.shape 

        all_mode_features = [] # store the features of all modalities
        
        x = self.ld_embedding(x)
        x = rearrange(x, "N T P C -> (N T) P C")
        x = self.spatial_encoding(x)
        x = rearrange(x, "(N T) P C -> (N P) T C", N=N)
        x = self.temporal_encoding_ld(x)
        x, _ = self.ld_temporal(x)
        x = self.ld_norm(x)
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
        # adjacency can be one or multiple adjacency matrices 
        if isinstance(metadata['adjacency'], torch.Tensor):
            metadata['adjacency'] = [metadata['adjacency']]
        adj_mtx = sum([s.detach() for s in metadata['adjacency']])
        binary_adjacency = (adj_mtx > 0)
        
        if isinstance(self.adjacency_hop, int) and self.adjacency_hop > 1:
            # do a loop instead of calling matrix power to avoid numerical problem
            for _ in range(self.adjacency_hop - 1): 
                binary_adjacency = torch.mm(binary_adjacency.float(), binary_adjacency.float())
                binary_adjacency = (binary_adjacency > 0)
                
        edge_index = torch.nonzero(binary_adjacency, as_tuple=False).T
        self.edge_index = edge_index

    def to(self, device: torch.device):
        if self.edge_index is not None:
            self.edge_index = self.edge_index.to(device)
        return super().to(device)
