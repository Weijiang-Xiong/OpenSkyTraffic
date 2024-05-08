import torch
import torch.nn as nn
import torch_geometric.nn as gnn

import numpy as np
from scipy.stats import rv_continuous, gennorm

from netsanut.loss import GeneralizedProbRegLoss
from typing import Dict, List, Tuple
from einops import rearrange

from netsanut.data.transform import TensorDataScaler
from netsanut.models.common import MLP_LazyInput, LearnedPositionalEncoding
from .catalog import MODEL_CATALOG

class HiMSNet(nn.Module):
    def __init__(self, use_drone=True, use_ld=True, use_global=True, normalize_input=True, scale_output=True, 
                 d_model=64, global_downsample_factor:int=2, layernorm=True, **kwargs):
        super().__init__()
        
        self.use_drone = use_drone
        self.use_ld = use_ld
        self.use_global = use_global
        if self.use_drone==False and self.use_ld==False:
            self.use_drone=True
            print("Must use at least one data modality, use drone data by default")
        self.normalize_input = normalize_input
        self.scale_output = scale_output
        
        self._calibrated_intervals: Dict[float, float] = dict()
        # self._beta: int = beta
        self._distribution: rv_continuous = None
        # self._set_distribution(beta=1)
        self.ignore_value = -1.0
        self.metadata: Dict[str, torch.Tensor] = None
        self.data_scalers: Dict[str, TensorDataScaler] = None

        self.spatial_encoding = LearnedPositionalEncoding(d_model=d_model, max_len=1570)
        
        if self.use_drone:
            self.drone_embedding = ValueEmbedding(d_model=d_model)
            self.temporal_encoding_drone = LearnedPositionalEncoding(d_model=d_model)
            # drone data has higher temporal resolution, so we use two conv layers to down sample
            self.drone_t_patching_1 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=3)
            self.drone_t_patching_2 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=3)
            self.drone_temporal = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=3, batch_first=True)
            self.drone_norm = nn.LayerNorm(d_model) if layernorm else nn.Identity()
            
        if self.use_ld:
            self.ld_embedding = ValueEmbedding(d_model=d_model)
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
            
        self.prediction = MLP_LazyInput(hid_dim=int(d_model * 2), out_dim=10, dropout=0.1)
        self.prediction_regional = MLP_LazyInput(hid_dim=128, out_dim=10, dropout=0.1)

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
                "drone_speed": self.data_scalers["drone_speed"].transform(data["drone_speed"].to(self.device)),
                "ld_speed": self.data_scalers["ld_speed"].transform(data["ld_speed"].to(self.device)),
            }
        else:
            source = {
                "drone_speed": data["drone_speed"].to(self.device),
                "ld_speed": data["ld_speed"].to(self.device),
            }
        target = {
            "pred_speed": data["pred_speed"].nan_to_num(nan=self.ignore_value).to(self.device),
            "pred_speed_regional": data["pred_speed_regional"].nan_to_num(nan=self.ignore_value).to(self.device),
        }
        
        return source, target

    def post_process(self, prediction: torch.Tensor) -> torch.Tensor:
        if self.scale_output:
            # scale back using the inverse_transform of the output scaler
            prediction["pred_speed"] = self.data_scalers["pred_speed"].inverse_transform(prediction["pred_speed"])
            prediction["pred_speed_regional"] = self.data_scalers["pred_speed_regional"].inverse_transform(
                prediction["pred_speed_regional"]
            )

        return prediction

    def make_prediction(self, source: dict[str, torch.Tensor]) -> torch.Tensor:
        x_drone, x_ld = source["drone_speed"], source["ld_speed"]
        N, T_drone, P, C = source["drone_speed"].shape
        T_ld = source["ld_speed"].shape[1]

        all_mode_features = [] # store the features of all modalities
        
        if self.use_drone:
            x_drone = self.drone_embedding(x_drone)
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
            x_ld = self.ld_embedding(x_ld)
            x_ld = rearrange(x_ld, "N T P C -> (N T) P C")
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
        
        regional_feature = torch.cat(
            [
                torch.mean(fused_features[:, self.metadata["cluster_id"] == region_id, :], dim=1).unsqueeze(1)
                for region_id in self.metadata["cluster_id"].unique()
            ],
            dim=1,
        )
        pred_regional = self.prediction_regional(regional_feature)

        return {
            "pred_speed": rearrange(pred, "N P T -> N T P"),
            "pred_speed_regional": rearrange(pred_regional, "N P T -> N T P"),
        }

    def compute_loss(self, source: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        pred_res = self.make_prediction(source)
        pred_res = self.post_process(pred_res) # scale back and then compute loss 
        loss = self.loss(pred_res["pred_speed"], target["pred_speed"])
        loss_regional = self.loss(pred_res["pred_speed_regional"], target["pred_speed_regional"])

        return {"loss": loss, "loss_regional": loss_regional}

    def inference(self, source: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.post_process(self.make_prediction(source))

    def adapt_to_metadata(self, metadata):
        self.metadata = dict()
        
        # keep the tensors and arrays on the same device as the model
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                self.metadata[key] = torch.as_tensor(value).to(self.device)
            elif isinstance(value, torch.Tensor):
                self.metadata[key] = value.to(self.device)
                
        self.data_scalers = {
            name: TensorDataScaler(
                mean=metadata["mean_and_std"][name][0], 
                std=metadata["mean_and_std"][name][1],
                data_dim=0,
                device=self.device
            ) for name in metadata["input_seqs"] + metadata["output_seqs"]
        }
        self.metadata["input_seqs"] = metadata["input_seqs"]
        self.metadata["output_seqs"] = metadata["output_seqs"]

    def _set_distribution(self, beta: int) -> rv_continuous:
        if beta is None:
            self._distribution = None
        else:
            self._distribution = gennorm(beta=int(beta))

        return self._distribution
    
    def state_dict(self):
        state = dict()
        state["model_params"] = super().state_dict()
        state["metadata"] = self.metadata
        state["data_scalers"] = {name: scaler.state_dict() for name, scaler in self.data_scalers.items()}
        return state

    def load_state_dict(self, state_dict):
        self.metadata = state_dict["metadata"]
        self.data_scalers = {
            name: TensorDataScaler(**state)
            for name, state in state_dict["data_scalers"].items()
        }
        super().load_state_dict(state_dict["model_params"])

class ValueEmbedding(nn.Module):
    
    def __init__(self, d_model:int, nan_place_holder=-1.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.time_emb_w = nn.Parameter(torch.randn(1, d_model))
        self.time_emb_b = nn.Parameter(torch.randn(1, d_model))
        self.value_emb_w = nn.Parameter(torch.randn(1, d_model))
        self.value_emb_b = nn.Parameter(torch.randn(1, d_model))
        self.empty_token = nn.Parameter(torch.randn(d_model)) # fit the tensor shape N, C, H, W
        self.unmonitored_token = nn.Parameter(torch.randn(d_model))
        
    def forward(self, x, empty_value = torch.nan, unmonitored: torch.Tensor = None):
        """_summary_

        Args:
            x (torch.Tensor): _description_
            empty_value (torch.Tensor): _description_. Defaults to torch.nan.
            unmonitored (torch.Tensor): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        N, T, P, C = x.shape
        value, time = x[:, :, :, 0], x[:, :, :, 1]
        
        time_emb = time.unsqueeze(-1) * self.time_emb_w + self.time_emb_b
        
        # comparing by == doesn't work with nan
        if empty_value in [float("nan"), np.nan, torch.nan, None]:
            empty_mask = torch.isnan(value)
        else:
            empty_mask = (value == empty_value)
        
        # replace the empty values (NaN) with the corresponding token
        value_emb = torch.empty(size=(N, T, P, self.d_model), device=self.device)
        value_emb[~empty_mask] = value.unsqueeze(-1)[~empty_mask] * self.value_emb_w+ self.value_emb_b
        value_emb[empty_mask] = self.empty_token
        
        # replace the unmonitored values using the unmonitored token
        if unmonitored is not None:
            value_emb[unmonitored] = self.unmonitored_token
            
        emb = time_emb + value_emb
        
        return emb.contiguous()

    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def extra_repr(self) -> str:
        return "d_model={}".format(self.d_model)

# one can write this as a decorator above the class definition, but that will lose the type hints 
# because in general one can not know what the decorator will return, so the type of the defined 
# model will be hinted as "Any", instead of the model class it belongs to
if __name__.endswith("himsnet"):
    MODEL_CATALOG.register(HiMSNet)
