import torch
import torch.nn as nn
import torch_geometric.nn as gnn

import numpy as np
from scipy.stats import rv_continuous, gennorm

from netsanut.loss import GeneralizedProbRegLoss
from typing import Dict, List, Tuple
from einops import rearrange

from netsanut.data.transform import TensorDataScaler
from .common import MLP_LazyInput, LearnedPositionalEncoding
from .catalog import MODEL_CATALOG

class GSwinFormer(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def adapt_to_metadata(self, metadata: Dict):
        pass 

if __name__.endswith("gswinformer"):
    MODEL_CATALOG.register(GSwinFormer)