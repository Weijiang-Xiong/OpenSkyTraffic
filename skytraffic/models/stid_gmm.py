from typing import Dict, List

import torch
from einops import rearrange

from .layers import GMMPredictionHead
from .stid import STIDNet


class STIDGMMNet(STIDNet):
    """
    STID network with a GMM prediction head in normalized space.
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
        anchors: List[float] = [-2.0, -1.0, 0.0, 1.0, 2.0],
        sizes: List[float] = [1.0, 1.0, 1.0, 1.0, 1.0],
        zero_init: bool = False,
        mcd_estimation: bool = False,
        input_steps: int = 12,
        pred_steps: int = 12,
        num_nodes: int = None,
    ):
        super().__init__(
            time_intervals=time_intervals,
            num_block=num_block,
            time_series_emb_dim=time_series_emb_dim,
            spatial_emb_dim=spatial_emb_dim,
            temp_dim_tid=temp_dim_tid,
            temp_dim_diw=temp_dim_diw,
            if_spatial=if_spatial,
            if_time_in_day=if_time_in_day,
            if_day_in_week=if_day_in_week,
            feature_dim=feature_dim,
            output_dim=output_dim,
            input_steps=input_steps,
            pred_steps=pred_steps,
            num_nodes=num_nodes,
        )

        del self.regression_layer
        self.prediction_head = GMMPredictionHead(
            in_dim=self.hidden_dim,
            hid_dim=self.hidden_dim * 2,
            anchors=anchors,
            sizes=sizes,
            pred_steps=pred_steps,
            dropout=0.15,  # Same dropout as in the original MLP
            zero_init=zero_init,
            mcd_estimation=mcd_estimation
        )

    def forward(self, source: torch.Tensor):
        hidden = self.feature_extraction(source)
        hidden = rearrange(hidden, "N C P 1 -> N P C")
        return self.prediction_head(hidden)

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
