import torch
import torch.nn as nn
from typing import List, Tuple
from einops import rearrange

import numpy as np

def gaussian_density(mean, log_var, x):
    """
    Compute Gaussian probability density in a numerically stable way using log variance.
    
    Args:
        mean: Mean of the Gaussian distribution
        log_var: Log of the variance (log σ²)
        x: Points at which to evaluate the density
        
    Returns:
        Gaussian probability density values
    """
    # Compute density using log variance for numerical stability
    return torch.exp(
        - 0.5 * ((x - mean) ** 2) / torch.exp(log_var) 
        - 0.5 * log_var 
        - 0.5 * torch.log(torch.tensor([2 * torch.pi], device=mean.device))
    )

class GMMPredictionHead(nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim: int,
        anchors: List[float],
        sizes: List[float],
        pred_steps: int = 10,
        loss_ignore_value: float = float("nan"),
        dropout=0.1,
        zero_init=True,
        mcd_estimation=False,
    ):
        super().__init__()
        self.pred_steps = pred_steps
        self.loss_ignore_value = loss_ignore_value
        self.zero_init = zero_init
        self.mcd_estimation = mcd_estimation # whether to use maximum a posteriori (MAP) estimation
        self.num_component = len(anchors)
        # the anchors are prioir knowledge for the mean 
        # put them as parameters so that they can be moved to the same device as the model
        self.anchors = nn.Parameter(torch.tensor(anchors).reshape(1, 1, 1, -1), requires_grad=False)
        self.sizes = nn.Parameter(torch.tensor(sizes).reshape(1, 1, 1, -1), requires_grad=False)
        self.size_scale = nn.Parameter(torch.zeros_like(self.sizes).reshape(1, 1, 1, -1), requires_grad=True)
         
        self.linear = nn.Linear(in_features=in_dim, out_features=hid_dim)
        self.norm = nn.LayerNorm(hid_dim)
        self.relu_act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # `mixing` coefficients of the Gaussian components, and `offset` with respect to the anchor. 
        # component_mean = anchor + dx * sizes * (1 + size_scale)
        # gmm_mean = (component_mean * mixing).mean()
        # each of them predicted for 10 future steps, so output 2x size in the deterministic case
        self.mixing = nn.Linear(in_features=hid_dim, out_features=pred_steps * len(anchors))
        self.offset = nn.Linear(in_features=hid_dim, out_features=pred_steps * len(anchors))
        self.uncertainty = nn.Linear(in_features=hid_dim, out_features=pred_steps * len(anchors))
    
        if self.zero_init:
            self.init_weights()

    def extra_repr(self):
        return f"dropout={self.dropout}, zero_init={self.zero_init}"
    
    @property
    def device(self):
        return next(self.parameters())[0].device
    
    def forward(self, x:torch.Tensor):
        # After feature extraction: x.shape=(N, P, C)
        x = self.linear(x)
        x = self.norm(x)
        x = self.relu_act(x)
        x = self.dropout(x)

        mixing = self.mixing(x)
        mixing = rearrange(mixing, "N P (T K) -> N P T K", T=self.pred_steps, K=self.num_component)
        mixing = torch.softmax(mixing, dim=-1)
        mixing = rearrange(mixing, "N P T K -> N T P K")
        
        offset = self.offset(x)
        offset = rearrange(offset, "N P (T K) -> N P T K", T=self.pred_steps, K=self.num_component)
        offset = rearrange(offset, "N P T K -> N T P K")
        means = self.anchors + offset * self.sizes * (1 + self.size_scale)
        
        log_var = self.uncertainty(x) # uncertainty in log scale
        log_var = rearrange(log_var, "N P (T K) -> N T P K", T=self.pred_steps, K=self.num_component)
        
        return mixing, means, log_var
    
    def losses(self, out:Tuple[torch.Tensor, torch.Tensor], target:torch.Tensor):
        """ `out` is the output of forward, and `target` is the label with shape (N, T, P)
        """
        mixing, means, log_var = out # all of shape (N, T, P, K) as returned in forward
        
        # Create mask for valid values (not NaN, not self.ignore_value) 
        valid_mask = ~target.isnan()
        # we don't use torch.isnan because it does not support python float numbers
        if not np.isnan(self.loss_ignore_value):
            valid_mask = torch.logical_and(valid_mask, target != self.loss_ignore_value)
        
        # Extract valid targets and corresponding predictions
        target = target[valid_mask]  # shape: (num_valid,)
        mixing = mixing[valid_mask]  # shape: (num_valid, K)
        means = means[valid_mask]    # shape: (num_valid, K)
        log_var = log_var[valid_mask]  # shape: (num_valid, K)
        
        # log probability for each component, since this is a loss, it is OK to drop the constant term
        #  - 0.5 * torch.log(torch.tensor(2 * torch.pi, device=target.device))
        log_prob = - 0.5 * (means - target.unsqueeze(-1)).pow(2) * torch.exp(-log_var) - 0.5 * log_var
        
        # add the mixing coefficient
        log_prob = log_prob + torch.log(mixing)
        
        # Combine the components 
        nll_loss = -torch.logsumexp(log_prob, dim=-1)
        
        # Return mean loss over valid samples
        nll_loss = nll_loss.mean()
        
        return {"nll_loss": nll_loss}
    
    def inference(self, out:Tuple[torch.Tensor, torch.Tensor]):
        """ `out` is the output of forward, and `target` is the label with shape (N, T, P)
        """
        mixing, means, log_var = out # all of shape (N, T, P, K) as returned in forward
        

        if self.mcd_estimation:
            # for a gaussian mixture model, the maximum a posteriori estimation can not be 
            # analytically solved, but since we formulate the prediction using anchors, the 
            # gaussian components should be somewhat separated, so we can use the component
            # with highest probability density at its own mean as the prediction
            # that is Maximum Component Denstiity (MCD) estimation
            max_index = torch.argmax(mixing*torch.exp(-log_var), dim=-1)
            pred = torch.gather(means, dim=-1, index=max_index.unsqueeze(-1)).squeeze(-1)
        else:
            pred = (mixing * means).sum(dim=-1)
        
        return pred, mixing, means, log_var

    def init_weights(self):
        # initialize the weights so that 
        # the predicted offset is close to zero
        # the predicted variance is close to 1
        # the predicted mixing coefficient is close to 1/num_component
        self.mixing.weight.data.fill_(0.0)
        self.mixing.bias.data.fill_(1/self.num_component)
        self.offset.weight.data.fill_(0.0)
        self.offset.bias.data.fill_(0.0)
        self.uncertainty.weight.data.fill_(0.0)
        self.uncertainty.bias.data.fill_(0.0)
    
    @staticmethod
    def get_mixture_density(mixing:torch.Tensor, means: torch.Tensor, log_var: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        """ Compute the probability density of a gaussian mixture model at points `xs`
        
        Notation of shape: 
            N - batch size
            T - time steps
            P - number of spatial locations
            K - number of Gaussian components
        
        Args:
            mixing (torch.Tensor): mixture coefficients, shape (N, T, P, K) 
            means (torch.Tensor): predicted means, shape (N, T, P, K)
            log_var (torch.Tensor): predicted log variance, shape (N, T, P, K)
            xs (torch.Tensor): the points to evaluate the probability density, shape (n_points,)

        Returns:
            mixture_density (torch.Tensor): shape (N, T, P, n_points)
        """
        
        component_densities = gaussian_density(
                            means[..., None], 
                            log_var[..., None], 
                            xs.reshape(*([1] * len(means.shape)), -1)) # xs[*[None]*len(means.shape), :]
        weighted_densities = mixing[..., None] * component_densities
        mixture_density = (weighted_densities).sum(axis=-2)
        
        return mixture_density

    @staticmethod
    def get_mixture_cdf(mixing, means, log_var, xs):
        gmm_density = GMMPredictionHead.get_mixture_density(mixing, means, log_var, xs)
        dx = abs(xs[1] - xs[0]) # the step size of the x values
        gmm_cdf = torch.cumsum(gmm_density * dx, axis=-1)
        gmm_cdf = gmm_cdf / gmm_cdf[..., -1].unsqueeze(-1) # ensure the CDF ends at 1
        
        return gmm_cdf

    @staticmethod
    def get_confidence_interval(mixing, means, log_var, xs, conf:float=0.90):
        """ Compute the confidence interval of a gaussian mixture model

        Args:
            mixing (torch.tensor): the mixing coefficients of the Gaussian components, shape (N, T, P, K)
            means (torch.tensor): the means of the Gaussian components, shape (N, T, P, K)
            log_var (torch.tensor): the log variance of the Gaussian components, shape (N, T, P, K)
            xs (torch.tensor): the points to evaluate the probability density, shape (n_points,)
            conf (float, optional): confidence level. Defaults to 0.90.

        Returns:
            lb (torch.tensor): lower bound of the confidence interval
            ub (torch.tensor): upper bound of the confidence interval
        """
        num_comp = mixing.shape[-1]
        xs = xs.to(mixing.device)
        dx = abs(xs[1] - xs[0])
        mixture_density = GMMPredictionHead.get_mixture_density(mixing, means, log_var, xs)

        values, indexes = torch.sort(mixture_density, dim=-1, descending=True)
        prob_mass = (values * dx).cumsum(dim=-1)
        # normalize the probability mass so that it sums to 1
        prob_mass = prob_mass / prob_mass[..., -1:]
        in_interval = prob_mass <= conf
        inv_index = torch.argsort(indexes, dim=-1)
        x_is_in_interval = torch.gather(in_interval, dim=-1, index=inv_index)
        # add a False at the beginning and end to cover the edge cases where the edge values are True. 
        # (T, T, F, F, T, T) => (F, T, T, F, F, T, T, F)
        x_is_in_interval = torch.cat([torch.zeros_like(x_is_in_interval[..., :1]).bool(), 
                                      x_is_in_interval, 
                                      torch.zeros_like(x_is_in_interval[..., :1]).bool()], dim=-1)
        
        diff = torch.diff(x_is_in_interval.int(), dim=-1)
        index_range = torch.arange(diff.size(-1), device=xs.device)
        # x_is_in_interval: (F, T, T, F, F, T, T, F); diff: [1, 0, -1, 0, 1, 0, -1], one element less.
        # the index of 1 in diff (0, 4) is the same as the lower bound index in x_is_in_interval (0, 4)
        # the index of -1 in diff (2, 6) is 1 + the index of the upper bound index in x_is_in_interval (1, 5)
        # We filled the other values with the biggest index (unpadded), so after sorting, they will be at 
        # the end then we can know the interval is empty by checking if lb == ub
        lb_index = torch.sort(
            torch.where(diff == 1, index_range, torch.full_like(index_range, fill_value=xs.size(-1)-1, device=xs.device)) 
            )[0][..., :num_comp]
        ub_index = torch.sort(
            torch.where(diff == -1, index_range-1, torch.full_like(index_range, fill_value=xs.size(-1)-1, device=xs.device))
            )[0][..., :num_comp]
        lb = xs[lb_index]
        ub = xs[ub_index]

        return lb, ub