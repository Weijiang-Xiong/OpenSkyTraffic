import torch

from typing import Dict, List


from .layers import GMMPredictionHead
from .lstmgcnconv import LSTMGCNConv

class LSTMGCNConv_GMM(LSTMGCNConv):
    def __init__(
        self,
        use_global=True,
        input_steps: int = 12,
        pred_steps: int = 12,
        num_nodes: int = None,
        d_model=64,
        global_downsample_factor: int = 1,
        layernorm=True,
        assume_clean_input: bool = True,
        adjacency_hop: int = 1,
        dropout: float = 0.1,
        anchors: List[float] = [-2.0, -1.0, 0.0, 1.0, 2.0],
        sizes: List[float] = [1.0, 1.0, 1.0, 1.0, 1.0],
        zero_init=True,
        mcd_estimation=False,
        metadata: dict = None,
    ):
        super().__init__(
            use_global=use_global,
            d_model=d_model,
            global_downsample_factor=global_downsample_factor,
            layernorm=layernorm,
            assume_clean_input=assume_clean_input,
            adjacency_hop=adjacency_hop,
            dropout=dropout,
            input_steps=input_steps,
            pred_steps=pred_steps,
            num_nodes=num_nodes,
            metadata=metadata,
        )
        del self.prediction
            
        # Replace MLP prediction head with GMMPredictionHead
        # Anchors and sizes designed for z-normalized data (mean=0, std=1)
        self.prediction_head = GMMPredictionHead(
            in_dim=d_model * int(1 + self.use_global),
            hid_dim=int(d_model * 2),
            anchors=anchors,
            sizes=sizes,
            pred_steps=pred_steps,
            dropout=dropout,
            zero_init=zero_init,
            mcd_estimation=mcd_estimation
        )

    def forward(self, source: torch.Tensor):
        fused_features = self.feature_extraction(source)
        return self.prediction_head(fused_features)

    def compute_loss(self, source: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        head_out = self(source)
        return self.prediction_head.losses(head_out, target)

    def inference(self, source: torch.Tensor) -> Dict[str, torch.Tensor]:
        head_out = self(source)
        pred, mixing, means, log_var = self.prediction_head.inference(head_out)

        return {
            "pred": pred,
            "mixing": mixing,
            "means": means,
            "log_var": log_var,
        }
