import torch
from typing import Dict, List
from einops import rearrange

from .staeformer import STAEformer
from .layers import GMMPredictionHead


class STAEformer_GMM(STAEformer):
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
        add_time_in_day: bool = True,
        add_day_in_week: bool = False,
        # GMM-specific parameters
        anchors: List[float] = [-2.0, -1.0, 0.0, 1.0, 2.0],
        sizes: List[float] = [1.0, 1.0, 1.0, 1.0, 1.0],
        zero_init: bool = False,
        mcd_estimation: bool = False,
        # dataset/task parameters
        input_steps: int = 12,
        pred_steps: int = 12,
        num_nodes: int = 207,
        metadata: dict = None,
    ):
        super().__init__(
            steps_per_day=steps_per_day,
            input_dim=input_dim,
            output_dim=output_dim,
            input_embedding_dim=input_embedding_dim,
            tod_embedding_dim=tod_embedding_dim,
            dow_embedding_dim=dow_embedding_dim,
            spatial_embedding_dim=spatial_embedding_dim,
            adaptive_embedding_dim=adaptive_embedding_dim,
            feed_forward_dim=feed_forward_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_mixed_proj=True,
            add_time_in_day=add_time_in_day,
            add_day_in_week=add_day_in_week,
            input_steps=input_steps,
            pred_steps=pred_steps,
            num_nodes=num_nodes,
            metadata=metadata,
        )
        
        # Remove the original prediction layer (assume use_mixed_proj is always True)
        del self.output_proj
        
        # Replace with GMM prediction head
        # Always use mixed projection dimension
        in_dim = self.input_steps * self.model_dim
        
        self.prediction_head = GMMPredictionHead(
            in_dim=in_dim,
            hid_dim=feed_forward_dim,
            anchors=anchors,
            sizes=sizes,
            pred_steps=pred_steps,
            dropout=dropout,
            zero_init=zero_init,
            mcd_estimation=mcd_estimation
        )

    def forward(self, source: torch.Tensor):
        x = self.feature_extraction(source)
        x = rearrange(x, "N T P C -> N P (T C)")
        return self.prediction_head(x)

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
