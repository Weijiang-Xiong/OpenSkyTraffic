import torch
from typing import Dict, List
from einops import rearrange

from .stid import STID
from .layers import GMMPredictionHead


class STID_GMM(STID):
    """
    Paper: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting
    Link: https://arxiv.org/abs/2208.05233
    Official Code: https://github.com/zezhishao/STID
    Adapted to use GMM prediction head for probabilistic forecasting
    """

    def __init__(
        self,
        # Model-specific parameters with defaults based on original implementation
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
        loss_ignore_value: float = float("nan"),
        norm_label_for_loss: bool = True,
        # GMM-specific parameters
        anchors: List[float] = [-2.0, -1.0, 0.0, 1.0, 2.0],
        sizes: List[float] = [1.0, 1.0, 1.0, 1.0, 1.0],
        zero_init: bool = False,
        mcd_estimation: bool = False,
        # BaseModel parameters
        input_steps: int = 12,
        pred_steps: int = 12,
        num_nodes: int = None,
        data_null_value: float = 0.0,
        metadata: dict = None,
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
            loss_ignore_value=loss_ignore_value,
            norm_label_for_loss=norm_label_for_loss,
            input_steps=input_steps,
            pred_steps=pred_steps,
            num_nodes=num_nodes,
            data_null_value=data_null_value,
            metadata=metadata,
        )
        
        # Remove the original regression layer
        del self.regression_layer
        
        # Replace with GMM prediction head
        self.prediction_head = GMMPredictionHead(
            in_dim=self.hidden_dim,
            hid_dim=self.hidden_dim * 2,
            anchors=anchors,
            sizes=sizes,
            pred_steps=pred_steps,
            dropout=0.15,  # Same dropout as in the original MLP
            loss_ignore_value=float("nan"),
            zero_init=zero_init,
            mcd_estimation=mcd_estimation
        )

    def make_predictions(self, source):
        hidden = self.feature_extraction(source)
        hidden = rearrange(hidden, "N C P 1 -> N P C")
        head_out = self.prediction_head(hidden)
        
        return head_out

    def compute_loss(self, source: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        head_out = self.make_predictions(source)
        loss = self.prediction_head.losses(head_out, target)
        return loss

    def inference(self, source: torch.Tensor) -> Dict[str, torch.Tensor]:
        head_out = self.make_predictions(source)
        pred, mixing, means, log_var = self.prediction_head.inference(head_out)
        
        res = {
            "pred": pred,
            "mixing": mixing,
            "means": means,
            "log_var": log_var
        }
        
        return self.post_process(res)

    def post_process(self, prediction: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        prediction['pred'] = self.datascaler.inverse_transform(prediction["pred"])
        prediction['mixing'] = prediction['mixing']
        prediction['means'] = self.datascaler.inverse_transform(prediction['means'])
        prediction['log_var'] = prediction['log_var'] + 2 * torch.log(self.datascaler.std)
        return prediction