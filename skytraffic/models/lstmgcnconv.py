import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from ..loss import masked_mae
from typing import Dict, Tuple
from einops import rearrange

from ..data.transform import TensorDataScaler
from .common import MLP, LearnedPositionalEncoding
from .catalog import MODEL_CATALOG
from .base import BaseModel

class LSTMGCNConv(BaseModel):
    def __init__(
        self,
        use_global=True,
        pred_steps: int = 12,
        d_model=64,
        global_downsample_factor: int = 1,
        layernorm=True,
        adjacency_hop: int = 1,
        dropout: float = 0.1,
        input_null_value: float = 0.0,
        norm_label_for_loss: bool = True,
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
        """
        super().__init__()
        self.use_global = use_global
        self.input_null_value = input_null_value
        self.adjacency_hop = adjacency_hop
        self.norm_label_for_loss = norm_label_for_loss
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
        self.loss = masked_mae

    def preprocess(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        source = data["source"].to(self.device)
        target = data["target"].to(self.device)

        # replace the label values with nan, so that they will be ignored in the loss after normalization
        target[target == self.input_null_value] = torch.nan

        # normalize the data
        source = self.datascaler.transform(source)
        if self.norm_label_for_loss:
            target = self.datascaler.transform(target, datadim_only=False)
        
        return source, target

    def post_process(self, prediction: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        prediction['pred'] = self.datascaler.inverse_transform(prediction["pred"])
        return prediction

    def compute_loss(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_res = self.make_prediction(source)
        # when label is scaled, we directly train the model to predict the scaled label
        # otherwise, we scale back the prediction and then compute the loss
        if self.norm_label_for_loss:
            loss = self.loss(pred_res["pred"], target)
        else:
            loss = self.loss(self.post_process(pred_res)["pred"], target)

        return {"loss": loss}

    def inference(self, source: torch.Tensor) -> torch.Tensor:
        return self.post_process(self.make_prediction(source))

    def make_prediction(self, source: dict[str, torch.Tensor]) -> torch.Tensor:
        fused_features = self.feature_extraction(source)
        pred = self.prediction(fused_features)
        return {
            "pred": rearrange(pred, "N P T -> N T P"),
        }

    def feature_extraction(self, x: torch.Tensor) -> torch.Tensor:
        N, _, P, C = x.shape 

        all_mode_features = [] # store the features of all modalities
        
        x = rearrange(x, "N T P C -> N C P T")
        x = self.ld_embedding(x)
        x = rearrange(x, "N C P T -> (N T) P C")
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
            x_inter = self.dropout(self.relu(self.global_norm1(self.gcn_1(x_global, self.metadata["edge_index"]))))
            x_inter = self.dropout(self.relu(self.global_norm2(self.gcn_2(x_inter, self.metadata["edge_index"]))))
            x_global = x_global + x_inter
            x_global = self.gcn_3(x_global, self.metadata["edge_index"])
            x_global = self.channel_up_sample(x_global)
            all_mode_features.append(x_global)

        fused_features = torch.cat(all_mode_features, dim=-1)
        
        return fused_features
    

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
    
    def state_dict(self):
        """ we add datascalar and metadata to the state_dict, so that they will be saved to the checkpoint, 
        and then can be loaded later.
        """
        state = dict()
        state["model_params"] = super().state_dict()
        state["metadata"] = self.metadata
        state["datascaler"] = self.datascaler.state_dict()
        return state
    
    def load_state_dict(self, state_dict, strict: bool = False):
        self.datascaler = TensorDataScaler(**state_dict["datascaler"])
        self.metadata = state_dict["metadata"]
        super().load_state_dict(state_dict["model_params"], strict=strict)
        
if __name__.endswith("lstmgcnconv"):
    MODEL_CATALOG.register(LSTMGCNConv)