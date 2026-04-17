from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from .base import BaseModel


class ForecastModel(nn.Module):
    """ Wrap a pure forecasting network with normalization.
    
        The input is usually a dict of tensors, and the shape is usually (N, T, P, C), where:
            N: number of samples
            T: number of time steps
            P: number of nodes
            C: number of features (e.g., 2 for the input, if the speed value is augmented with time in day)
    """

    def __init__(
        self,
        model: BaseModel,
        normalizer: nn.Module,
        data_null_value: float,
        metadata: dict = None,
    ):
        super().__init__()
        self.model = model
        self.normalizer = normalizer
        self.data_null_value = data_null_value

        if metadata is not None:
            self.adapt_to_metadata(metadata)


    def forward(self, data: dict[str, torch.Tensor]):
        """
        Main forward pass that handles both training and inference modes.
        
        Args:
            data: Dictionary containing input data and optionally target data
            
        Returns:
            Dict containing loss values (training) or predictions (inference)
        """
        # preprocessing (if any)
        source, target = self.preprocess(data)
        
        if self.training:
            assert target is not None, "target should be provided for training"
            return self.compute_loss(source, target)
        else:
            return self.inference(source)


    def preprocess(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor | None]:
        source = data["source"].to(self.model.device).clone()
        target = data.get("target")

        if target is not None:
            target = target.to(self.model.device).clone()
            if not np.isfinite(self.data_null_value):
                target[target.isnan()] = float("nan")
            else:
                target[target == self.data_null_value] = float("nan")
            target = self.normalizer.transform(target, datadim_only=False)

        source = self.normalizer.transform(source)
        return source, target

    def compute_loss(self, source: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model.compute_loss(source, target)

    def inference(self, source: torch.Tensor) -> Dict[str, torch.Tensor]:
        prediction = self.model.inference(source)
        prediction = self.normalizer.inverse_transform(prediction)
        if isinstance(prediction, dict):
            return prediction
        return {"pred": prediction}

    def adapt_to_metadata(self, metadata):
        self.model.adapt_to_metadata(metadata)
        self.normalizer.adapt_to_metadata(metadata)

    def to(self, device: torch.device | str):
        super().to(device)
        self.model.to(device)
        self.normalizer.to(device)
        return self
