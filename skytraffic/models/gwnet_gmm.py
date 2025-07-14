import torch
from typing import Dict, List
from einops import rearrange

from .gwnet import GWNET
from .layers import GMMPredictionHead


class GWNET_GMM(GWNET):
    def __init__(
        self,
        # Model-specific parameters with defaults based on original implementation
        dropout: float = 0.3,
        blocks: int = 4,
        layers: int = 2,
        gcn_bool: bool = True,
        addaptadj: bool = True,
        randomadj: bool = True,
        aptonly: bool = True,
        kernel_size: int = 2,
        nhid: int = 32,
        residual_channels: int = None,
        dilation_channels: int = None,
        skip_channels: int = None,
        end_channels: int = None,
        apt_layer: bool = True,
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
            dropout=dropout,
            blocks=blocks,
            layers=layers,
            gcn_bool=gcn_bool,
            addaptadj=addaptadj,
            randomadj=randomadj,
            aptonly=aptonly,
            kernel_size=kernel_size,
            nhid=nhid,
            residual_channels=residual_channels,
            dilation_channels=dilation_channels,
            skip_channels=skip_channels,
            end_channels=end_channels,
            apt_layer=apt_layer,
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
            loss_ignore_value=loss_ignore_value,
            zero_init=zero_init,
            mcd_estimation=mcd_estimation
        )

    def make_predictions(self, source: torch.Tensor):
        x = self.feature_extraction(source)
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