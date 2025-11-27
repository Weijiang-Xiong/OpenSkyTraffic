from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from skymonitor.patch_lgc import PatchedMVLSTMGCNConv


class TrafficPredictor:
    """ Basically a wrapper class for the neural network, the RL environment gives observation per-time-step (3 mins),
        but the predictor will need a context window (e.g., 30 mins) to make predictions.
        Not sure what's needed for this class, but leave the interface here just in case the input or output needs 
        some format handling.
    """

    def __init__(self, device: Optional[torch.device] = "cpu"):
        self.device = torch.device(device)
        self.net: nn.Module = PatchedMVLSTMGCNConv(
            use_global=True,
            feature_dim=3,
            d_model=64,
            temp_patching=3,
            global_downsample_factor=1,
            layernorm=True,
            adjacency_hop=1,
            dropout=0.1,
            loss_ignore_value=float("nan"),
            norm_label_for_loss=True,
            input_steps=360,
            pred_steps=10,
            num_nodes=1570,
            pred_feat=2,
            data_null_value=0.0,
        )
        state_dict = torch.load("./scratch/patch_lgc_simbarca_explore/model_final.pth")
        self.net.load_state_dict(state_dict['model'])
        self.net.eval()
        self.net.to(self.device)
        self.in_steps = self.net.input_steps
        self.output_steps = self.net.pred_steps

    def predict(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:

        predictions = self.net(data_dict)

        return predictions

    def __call__(self, new_data: Dict):
        return self.predict(new_data)
