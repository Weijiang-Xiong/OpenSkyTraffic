"""Minimal base interface for pure forecasting networks."""

from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    """
    Minimal base class for pure forecasting networks.

    Concrete subclasses implement the neural network forward pass and the loss in model space.
    Data preprocessing and label normalization belong in outer wrappers such as ForecastModelWrapper.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def device(self):
        """Get the device of the model parameters."""
        param = next(self.parameters(), None)
        return param.device if param is not None else torch.device("cpu")

    @property
    def num_params(self):
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @abstractmethod
    def compute_loss(self, source, target) -> Dict[str, torch.Tensor]:
        """
        Compute the training loss from already-preprocessed model inputs.
        """
        raise NotImplementedError

    def inference(self, source):
        """
        Run inference in model space.

        Deterministic models can usually use the raw forward pass directly.
        """
        return self(source)

    def adapt_to_metadata(self, metadata):
        """
        Optional hook for models that need dataset metadata.
        """
        return self
