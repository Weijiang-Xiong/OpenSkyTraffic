""" This file defines a base model interface and 
    shared functionalities for neural network models
"""
import logging 
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Mapping

import torch 
import torch.nn as nn 

logger = logging.getLogger("default")

class BaseModel(nn.Module, ABC):
    """
    Abstract base class for neural network models in the SkyTraffic project.
    
    Provides common interface and functionality that all models should implement,
    including data preprocessing, training/inference logic, and metadata handling.
    
    The input tensor is assumed to have shape (N, T, P, C), where:
        N: number of samples
        T: number of time steps
        P: number of nodes
        C: number of features (e.g., 2 for the input, if the speed value is augmented with time in day)
    
    Args:
        input_steps: the number of input time steps
        pred_steps: the number of prediction time steps
        num_nodes: the number of nodes in the graph, commonly required for encoding
        adjacency: the adjacency matrix of the graph, commonly required for graph-based models
        data_null_value: the null value in the data, which indicates missing values
        metadata: the metadata of the dataset, allows more customizations, e.g., constructing a different adjacency matrix
        
    """

    def __init__(
        self,
        input_steps: int = None,
        pred_steps: int = None,
        num_nodes: int = None,
        data_null_value: float = None,
        metadata: dict = None,
    ) -> None:
        super().__init__()
        self.input_steps = input_steps if input_steps is not None else metadata["input_steps"]
        self.pred_steps = pred_steps if pred_steps is not None else metadata["pred_steps"]
        self.num_nodes = num_nodes if num_nodes is not None else metadata["num_nodes"]
        self.adjacency = metadata.get("adjacency", None) if isinstance(metadata, Mapping) else None
        self.data_null_value = data_null_value if data_null_value is not None else metadata["data_null_value"]

    @property   
    def device(self):
        """Get the device of the model parameters"""
        return list(self.parameters())[0].device if len(list(self.parameters())) > 0 else torch.device("cpu")
    
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

    def adapt_to_metadata(self, metadata):
        """
        In traffic analysis tasks, the model often needs the adjacency matrix to describe
        the graph structure, and it may also need the mean and std of the data to normalize them.
        This method is a recommended wrapper function to be called when initializing the model,
        it is not mandatory as one can explicitly write the codes in the __init__ function.
        
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
    
    def to(self, device: torch.device | str):
        """
        Move the required tensors to the specified device.
        In addition to the model parameters, we usually need to move adjacency matrix, edge indexes or data scalars, etc.
        But what to move eactly will depend on the specific model, and threfore is not implemented in the base class.
        """
        if isinstance(device, str):
            device = torch.device(device)
        return super().to(device)