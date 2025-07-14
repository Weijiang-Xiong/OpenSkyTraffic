import torch
import torch.nn.functional as F
from typing import Dict, List
from einops import rearrange

from .mtgnn import MTGNN
from .layers import GMMPredictionHead


class MTGNN_GMM(MTGNN):
    def __init__(
        self,
        # Model-specific parameters with defaults based on original implementation
        gcn_true: bool = True,
        buildA_true: bool = True,
        gcn_depth: int = 2,
        dropout: float = 0.3,
        subgraph_size: int = 20,
        node_dim: int = 40,
        dilation_exponential: int = 1,
        conv_channels: int = 32,
        residual_channels: int = 32,
        skip_channels: int = 64,
        end_channels: int = 128,
        layers: int = 3,
        propalpha: float = 0.05,
        tanhalpha: float = 3,
        layer_norm_affline: bool = True,
        use_curriculum_learning: bool = False,
        step_size: int = 2500,
        max_epoch: int = 100,
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
            gcn_true=gcn_true,
            buildA_true=buildA_true,
            gcn_depth=gcn_depth,
            dropout=dropout,
            subgraph_size=subgraph_size,
            node_dim=node_dim,
            dilation_exponential=dilation_exponential,
            conv_channels=conv_channels,
            residual_channels=residual_channels,
            skip_channels=skip_channels,
            end_channels=end_channels,
            layers=layers,
            propalpha=propalpha,
            tanhalpha=tanhalpha,
            layer_norm_affline=layer_norm_affline,
            use_curriculum_learning=use_curriculum_learning,
            step_size=step_size,
            max_epoch=max_epoch,
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
        
        # Remove the original prediction layers
        del self.end_conv_2
        
        # Replace with GMM prediction head
        self.prediction_head = GMMPredictionHead(
            in_dim=self.end_channels,
            hid_dim=self.end_channels,
            anchors=anchors,
            sizes=sizes,
            pred_steps=pred_steps,
            dropout=dropout,
            loss_ignore_value=float("nan"),
            zero_init=zero_init,
            mcd_estimation=mcd_estimation
        )

    def make_predictions(self, source: torch.Tensor, idx=None):
        x = self.feature_extraction(source, idx)
        x = rearrange(x, "N C P 1 -> N P C")
        head_out = self.prediction_head(x)
        
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