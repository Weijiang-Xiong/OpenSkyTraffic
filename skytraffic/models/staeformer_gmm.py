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
        num_nodes: int = 207,
        data_null_value: float = 0.0,
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
            loss_ignore_value=loss_ignore_value,
            norm_label_for_loss=norm_label_for_loss,
            input_steps=input_steps,
            pred_steps=pred_steps,
            num_nodes=num_nodes,
            data_null_value=data_null_value,
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
            loss_ignore_value=float("nan"),
            zero_init=zero_init,
            mcd_estimation=mcd_estimation
        )

    def make_predictions(self, source):
        """
        Make predictions using GMM head.
        """
        # Get features from the parent class feature_extraction
        x = self.feature_extraction(source)
        x = rearrange(x, "N T P C -> N P (T C)")
        # Get GMM predictions
        head_out = self.prediction_head(x)
        
        return head_out

    def compute_loss(self, source: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss during training.
        """
        # Get GMM predictions
        head_out = self.make_predictions(source)
        # Compute loss
        loss = self.prediction_head.losses(head_out, target)
        return loss

    def inference(self, source: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform inference/prediction.
        """
        # Get GMM predictions
        head_out = self.make_predictions(source)
        # Get predictions
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