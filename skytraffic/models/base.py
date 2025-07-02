""" This file defines a base model interface and 
    shared functionalities for neural network models
"""
import logging 
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

import torch 
import torch.nn as nn 

logger = logging.getLogger("default")

class BaseModel(nn.Module, ABC):
    """
    Abstract base class for neural network models in the SkyTraffic project.
    
    Provides common interface and functionality that all models should implement,
    including data preprocessing, training/inference logic, and metadata handling.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.metadata = None

    @property   
    def device(self):
        """Get the device of the model parameters"""
        return list(self.parameters())[0].device
    
    @property
    def num_params(self):
        """Get the number of trainable parameters"""
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

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

    @abstractmethod
    def preprocess(self, data: dict[str, torch.Tensor]) -> Tuple[Any, Any]:
        """
        Preprocess input data before feeding to the model.
        
        Args:
            data: Raw input data dictionary
            
        Returns:
            Tuple of (source, target) where source is preprocessed input
            and target is the ground truth (can be None during inference)
        """
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, source: Any, target: Any) -> Dict[str, torch.Tensor]:
        """
        Compute loss during training.
        
        Args:
            source: Preprocessed input data
            target: Ground truth data
            
        Returns:
            Dictionary containing loss values (must include 'loss' key)
        """
        raise NotImplementedError
    
    @abstractmethod
    def inference(self, source: Any) -> Dict[str, torch.Tensor]:
        """
        Perform inference/prediction.
        
        Args:
            source: Preprocessed input data
            
        Returns:
            Dictionary containing model predictions
        """
        raise NotImplementedError

    @abstractmethod
    def adapt_to_metadata(self, metadata):
        """
        Adapt the model to dataset-specific metadata.
        
        This method is called during training setup to configure the model
        based on dataset characteristics like normalization parameters,
        adjacency matrices, etc.
        
        Args:
            metadata: Dataset metadata dictionary
        """
        raise NotImplementedError

    def state_dict(self):
        """
        Get model state dictionary.
        
        Default implementation returns standard PyTorch state dict.
        Override if you need to save additional state (metadata, scalers, etc.)
        """
        return super().state_dict()
        
    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Load model state dictionary.
        
        Default implementation uses standard PyTorch loading.
        Override if you saved additional state in state_dict().
        """
        return super().load_state_dict(state_dict, strict)