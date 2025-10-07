""" Notation of shapes:
        N: batch size
        T: sequence length (time steps)
        C: channel dimension, feature dimension
        M: number of nodes (sensors)

    Assumed input shape: (N, T, M, C_feat), where C_feat is the feature dimension

    Output shape: (N, T, M, C_out), where C_out is out dimension, squeezed to (N, T, M) if C_out is 1
"""
import copy
import logging

import torch
import torch.nn as nn

from typing import List, Tuple, Dict, Callable, Any, Mapping
from scipy.stats import rv_continuous, gennorm
from einops import rearrange

from .layers import GeneralizedProbRegLoss
from .utils.transform import TensorDataScaler
from .layers import LearnedPositionalEncoding, PositionalEncoding

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger("default")

class DividedTimeSpaceAttentionLayer(nn.Module):

    """
        Compute attention separately on the space dimension and time dimension, set `time_first=True` to compute first for time dimension, otherwise space dimension goes first.

        Assumed input and output shape: (N, C, T, M), so it can be stacked without shape manipulation in between.
    """

    def __init__(self, hid_dim, nhead, ff_dim, dropout, time_first=True) -> None:
        super().__init__()
        self.time_first = time_first
        self.space_attention = nn.TransformerEncoderLayer(
            d_model=hid_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.temporal_attention = nn.TransformerEncoderLayer(
            d_model=hid_dim, 
            nhead=nhead, 
            dim_feedforward=ff_dim, 
            dropout=dropout, 
            batch_first=True
        )

    def forward(self, x:torch.Tensor, s_mask=None, t_mask=None) -> torch.Tensor:
        N, T, M, C = x.shape 
        if self.time_first:
            x = rearrange(x, 'N T M C -> (N M) T C')
            x = self.temporal_attention(x, t_mask)
            x = rearrange(x, '(N M) T C -> (N T) M C', M=M)
            x = self.space_attention(x, s_mask)
            x = rearrange(x, '(N T) M C -> N T M C', N=N)
        else:
            x = rearrange(x, 'N T M C -> (N T) M C')
            x = self.space_attention(x, s_mask)
            x = rearrange(x, '(N T) M C -> (N M) T C', T=T)
            x = self.temporal_attention(x, t_mask)
            x = rearrange(x, '(N M) T C -> N T M C', N=N)
        return x 

class SpatialTemporalEncoder(nn.Module):
    
    def __init__(self, num_layers, hid_dim, nhead, ff_dim, dropout, time_first=True) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DividedTimeSpaceAttentionLayer(
                hid_dim,
                nhead,
                ff_dim,
                dropout,
                time_first
            ) for _ in range(num_layers)])
        
    def forward(self, x, s_mask=None, t_mask=None) -> torch.Tensor:
        layer: DividedTimeSpaceAttentionLayer
        for layer in self.layers:
            x = layer(x, s_mask, t_mask)
        return x 



class TemporalAggregate(nn.Module):
    
    def __init__(self, in_dim:int, out_dim:int=1, mode='linear') -> None:
        super().__init__()
        self.in_dim = in_dim
        self.mode = mode
        assert mode in ['linear', 'last', 'avg']
        match mode:
            case 'linear':
                self.agg = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=(1,1))
            case 'last':
                self.agg = lambda x: x[:, -1, ...]
            case 'avg':
                self.agg = lambda x: torch.mean(x, 1)
                
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        
        # 'N T M C -> N M C'
        x = self.agg(x)
        
        # only squeeze the time dimension, as N could be 1 in the last test batch
        return x.squeeze(1)
    
    def extra_repr(self) -> str:
        return "in_dim={}, mode={}".format(self.in_dim, self.mode)
        
class SpatialDecoder(nn.Module):
    """ stack a few transformer decoder layers, implementation is different from nn.TransformerDecoder
    """
    def __init__(self, num_layers, hid_dim, nhead, ff_dim, dropout) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(
            d_model=hid_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        ) for _ in range(num_layers)])
        
    def forward(self, query, memory, mask) -> torch.Tensor:
        layer: nn.TransformerDecoderLayer
        x = query
        for layer in self.layers:
            x = layer(x, memory, mask)
        return x 

def get_trivial_forward() -> Callable:
    
    def trivial_forward(x: torch.Tensor):
        return x
    
    return copy.deepcopy(trivial_forward)

class NeTSFormer(nn.Module):

    """ Networked Time Series Prediction with Transformer

        The idea resembles the divided space-time attention in TimesFormer for video action recognition, see https://github.com/facebookresearch/TimeSformer
    """

    def __init__(self,
                 in_dim: int = 2,
                 hid_dim: int = 64,
                 ff_dim: int = 256,
                 hist_len: int = 12,
                 pred_len: int = 12,
                 nhead: int = 2,
                 dropout: int = 0.1,
                 encoder_layers: int = 2,
                 decoder_layers: int = 2,
                 time_first: bool = True,
                 temp_aggregate: str = "avg",
                 auxiliary_metrics: bool = True, 
                 reduction: str = "mean",  # loss related
                 aleatoric: bool = False,
                 exponent: int = 1,
                 alpha: float = 1.0,
                 ignore_value: float = 0.0,
                 se_type: str = "learned", # encoding related
                 se_init: str = "rand",
                 te_type: str = "fixed",
                 te_init: str = "",
                 temp_causal: bool = False, # add causal mask to temporal attention
                ) -> None:

        super().__init__()
        
        # Initialize attributes from BaseModel
        self.datascaler: TensorDataScaler
        self._calibrated_intervals: Dict[float, float] = dict() 
        self._beta: int = exponent
        self._distribution: rv_continuous = None
        self._set_distribution(beta=self._beta)
        
        self.sto_layers = list()
        
        self.feature_embedding = nn.Conv2d(in_dim, hid_dim, kernel_size=(1, 1))
        self.spatial_encoding     = self.build_encoding_layer(hid_dim, dropout, se_type, se_init)
        self.temporal_encoding    = self.build_encoding_layer(hid_dim, dropout, te_type, te_init)
        
        self.encoder = SpatialTemporalEncoder(
            num_layers=encoder_layers,
            hid_dim=hid_dim,
            nhead=nhead,
            ff_dim=ff_dim,
            dropout=dropout,
            time_first=time_first
        )
        self.temporal_aggregate = TemporalAggregate(in_dim=hist_len, mode=temp_aggregate) # reduce time dimension
        self.decoder = SpatialDecoder(
            num_layers=decoder_layers,
            hid_dim=hid_dim,
            nhead=nhead,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        
        self.pred_mean = nn.Linear(in_features=hid_dim, out_features=pred_len)
        self.pred_var = nn.Linear(in_features=hid_dim, out_features=pred_len)
        self.sto_layers.append("pred_var")
        
        self.datascaler: TensorDataScaler
        self.loss = GeneralizedProbRegLoss(
            reduction=reduction,
            aleatoric=aleatoric,
            exponent=exponent,
            alpha=alpha,
            ignore_value=ignore_value
        )
        self.register_buffer(
            'temporal_mask',
            torch.logical_not(
                torch.ones(hist_len, hist_len, dtype=torch.bool).tril(diagonal=0)
                ).to(self.device) if temp_causal else None
        ) # use register buffer so it will be moved to the correct device by model.to(device)

    @property   
    def device(self):
        return list(self.parameters())[0].device
    
    @property
    def num_params(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])
    
    @property
    def is_probabilistic(self):
        return self._distribution is not None

    def forward(self, data: dict[str, torch.Tensor]):
        """
            time series forecasting task, 
            data is assumed to have (N, T, M, C) shape (assumed to be unnormalized)
            label is assumed to have (N, T, M) shape (assumed to be unnormalized)
            
            compute loss in training mode, predict future values in inference
        """
        
        # preprocessing (if any)
        source, target = self.preprocess(data)
        
        if self.training:
            assert target is not None, "label should be provided for training"
            return self.compute_loss(source, target)
        else:
            # we should not have target sequences in inference
            return self.inference(source)

    def preprocess(self, data: dict[str, torch.Tensor]) -> Tuple[Any, Any]:
        
        source = self.datascaler.transform(data['source'].to(self.device))
        target = data['target'].to(self.device)
        
        return source, target

    def _set_distribution(self, beta:int) -> rv_continuous:
        
        if beta is None:
            self._distribution = None 
        else:
            self._distribution = gennorm(beta=int(beta))

        return self._distribution
    
    def offset_coeff(self, 
                    confidence:float, 
                    return_calibrated=True
        )-> Dict[float, float]:
        """ 
        compute a `k` for a Generalized Gaussian Distribution parameterized by `beta`
        such that 0.5 * confidence_interval_width = k * std, 
        one can later obtain the confidence interval by
            upper_bound = mean + k * pred_std
            lower_bound = mean - k * pred_std
            
        References
            https://en.wikipedia.org/wiki/Generalized_normal_distribution
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gennorm.html

        Args:
            confidence (float): _description_
            return_calibrated (bool, optional): _description_. Defaults to True.

        Returns:
            Dict[float,Tuple[float, float]]: a mapping from confidence score to offset coefficient
        """
        # use the calibrated result if it has been calibrated 
        if return_calibrated and confidence in self._calibrated_intervals.keys():
            interval = self._calibrated_intervals[confidence]
        else:
            try:
                interval = self._distribution.interval(confidence)
            except AttributeError:
                logger.warning("Make sure self._distribution is properly initialized, and it has the method `interval` for confidence intervals")
                interval = (0.0, 0.0)
            
        return abs(interval[0])
    
    def post_process(self, res):
        # the scale of uncertainty, supposed to have the same unit as prediction
        # so it makes sense to add them. 
        res['sigma'] = torch.exp(res['plog_sigma']*(1.0/self._beta))
        return res
    
    def update_calibration(self, intervals: Dict[float,Tuple[float, float]]):
        self._calibrated_intervals.update(intervals)

    def make_pred(self, x:torch.Tensor) -> Tuple[torch.Tensor]:
        """ 
            run a forward pass through the network 
            data is of shape (N, T, M, C_in), assumed to be normalized 
            output shape (N, T, M)
        """
        N, T, M, C_in = x.shape
        x = rearrange(x, 'N T M C -> N C M T')
        x = self.feature_embedding(x)
        x = rearrange(x, 'N C M T -> (N M) T C')
        x = self.temporal_encoding(x)
        x = rearrange(x, '(N M) T C -> (N T) M C', M=M)
        x = self.spatial_encoding(x)
        x = rearrange(x, '(N T) M C -> N T M C', N=N)
        x = self.encoder(x, t_mask=self.temporal_mask)
        x = self.temporal_aggregate(x) # 'N T M C -> N M C'
        x = self.decoder(x, x, self.mask) # 'N M C -> N M C'
        mean = self.pred_mean(x) # 'N M C -> N M T'
        mean = rearrange(mean, 'N M T -> N T M')
        plog_sigma = self.pred_var(x)
        plog_sigma = rearrange(plog_sigma, 'N M T -> N T M')
        
        return mean, plog_sigma

    
    def inference(self, data, label=None) -> torch.Tensor:
        
        """ if label is None, this is basically the same as self.make_pred 
            one can use this as an alternative to writing torch.no_grad outside the model 
            if label is provided, auxiliary metrics will be computed according to the flag `self.record_auxiliary_metrics`
        """
        
        mean, plog_sigma = self.make_pred(data)
            
        mean = self.datascaler.inverse_transform(mean)
        
        res = {"pred": mean, "plog_sigma": plog_sigma}
        
        return self.post_process(res)
        
        
    def compute_loss(self, data, label) -> Dict[str, torch.Tensor]:
        
        mean, plog_sigma = self.make_pred(data)
        
        # scale back the predictions and then calculate loss 
        mean = self.datascaler.inverse_transform(mean)
        loss = self.loss(mean, label, plog_sigma)
            
        loss_dict = {"loss": loss}
            
        return loss_dict
    
    
    def adapt_to_metadata(self, metadata) -> None:
        
        self.datascaler = TensorDataScaler(mean=metadata['mean'], std=metadata['std'])
        # adjacency can be one or multiple adjacency matrices 
        if not isinstance(metadata['adjacency'], (list, tuple)):
            metadata['adjacency'] = [metadata['adjacency']]
        mask = sum([s.detach() for s in metadata['adjacency']])
        # a True value in self.mask indicates the corresponding key will be ignored
        self.mask = (mask == 0).to(self.device)
    
    @staticmethod 
    def build_encoding_layer(hid_dim, dropout, encoding_type:str, init_method="rand") -> nn.Module | Callable:
        match encoding_type:
            case "learned":
                return LearnedPositionalEncoding(hid_dim, dropout, init_method=init_method)
            case "fixed":
                return PositionalEncoding(hid_dim, dropout)
            case "None" | "none" | "" | None:
                return get_trivial_forward()
            
    def get_param_groups(self) -> Dict[str, List[nn.Parameter]]:
        
        det_params, sto_params = [], []
        
        for m_name, module in self.named_children():
            for p_name, param in module.named_parameters(remove_duplicate=True, recurse=True):
                if not param.requires_grad:
                    continue
            
                if m_name in self.sto_layers:
                    sto_params.append(("{}.{}".format(m_name, p_name), param))
                else:
                    det_params.append(("{}.{}".format(m_name, p_name), param))
        
        
        return {"det": [param for _, param in det_params],
                "sto": [param for _, param in sto_params]}
    
    def state_dict(self):
        ret = super().state_dict()
        ret["_beta"] = self._beta
        ret["_calibrated_intervals"] = self._calibrated_intervals
        return ret
        
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = False):
        self._beta = state_dict["_beta"]
        self._set_distribution(self._beta)
        self._calibrated_intervals = state_dict["_calibrated_intervals"]
        del state_dict["_beta"]
        del state_dict["_calibrated_intervals"]
        super().load_state_dict(state_dict, strict)
        return