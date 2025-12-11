from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from skytraffic.config import LazyConfig, instantiate

class TrafficPredictor:
    """ Basically a wrapper class for the neural network, the RL environment gives observation per-time-step (3 mins),
        but the predictor will need a context window (e.g., 30 mins) to make predictions.
        Not sure what's needed for this class, but leave the interface here just in case the input or output needs 
        some format handling.
    """

    def __init__(self, device: Optional[torch.device] = "cpu", ckpt_dir: Optional[str] = None):
        self.device = torch.device(device)
        config = LazyConfig.load(f"{ckpt_dir}/config.yaml")
        self.net = instantiate(config.model)
        state_dict = torch.load(f"{ckpt_dir}/model_final.pth")
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
