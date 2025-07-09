import torch

from typing import Dict, List, Any

from .catalog import MODEL_CATALOG
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
        adjacency_hop: int = 1,
        dropout: float = 0.1,
        data_null_value: float = 0.0,
        loss_ignore_value = float("nan"),
        norm_label_for_loss: bool = True,
        anchors: List[float] = [-2.0, -1.0, 0.0, 1.0, 2.0],
        sizes: List[float] = [1.0, 1.0, 1.0, 1.0, 1.0],
        zero_init=False,
        mcd_estimation=False,
        metadata: dict = None,
    ):
        super().__init__(
            use_global=use_global,
            d_model=d_model,
            global_downsample_factor=global_downsample_factor,
            layernorm=layernorm,
            adjacency_hop=adjacency_hop,
            dropout=dropout,
            data_null_value=data_null_value,
            loss_ignore_value=loss_ignore_value,
            norm_label_for_loss=norm_label_for_loss,
            input_steps=input_steps,
            pred_steps=pred_steps,
            num_nodes=num_nodes,
            metadata=metadata,
        )
        del self.prediction
        del self.loss
            
        # Replace MLP prediction head with GMMPredictionHead
        # Anchors and sizes designed for z-normalized data (mean=0, std=1)
        self.prediction_head = GMMPredictionHead(
            in_dim=d_model * int(1 + self.use_global),
            hid_dim=int(d_model * 2),
            anchors=anchors,
            sizes=sizes,
            pred_steps=pred_steps,
            dropout=dropout,
            loss_ignore_value=float("nan"),
            zero_init=zero_init,
            mcd_estimation=mcd_estimation
        )

    def preprocess(self, data):
        return super().preprocess(data)
    
    def feature_extraction(self, source: torch.Tensor) -> torch.Tensor:
        return super().feature_extraction(source)
    
    def compute_loss(self, source: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss during training.
        
        Args:
            source: Preprocessed input data
            target: Ground truth data containing target sequences
            
        Returns:
            Dictionary containing loss values
        """
        # Extract features
        fused_features = self.feature_extraction(source)
        # Get GMM predictions
        head_out = self.prediction_head(fused_features)
        # Compute loss
        loss = self.prediction_head.losses(head_out, target)

        return loss

    def inference(self, source: Any) -> Dict[str, torch.Tensor]:
        """
        Perform inference/prediction.
        
        Args:
            source: Preprocessed input data
            
        Returns:
            Dictionary containing model predictions
        """
        # Extract features
        fused_features = self.feature_extraction(source)
        # Get GMM predictions
        head_out = self.prediction_head(fused_features)
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

        
if __name__.endswith("lstmgcnconv_gmm"):
    MODEL_CATALOG.register(LSTMGCNConv_GMM) 