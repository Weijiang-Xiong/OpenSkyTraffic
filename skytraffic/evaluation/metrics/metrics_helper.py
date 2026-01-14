import torch
import numpy as np
from typing import List, Tuple

from .probabilistic import interval_coverage_and_width, crps_from_cdf
from ...models.layers import GMMPredictionHead


def get_point_cdf(tensor: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """ tensor has shape (N, T, sp_size), can be ground-truth or predicted most-likely values.
        xs has shape (X,), which is the points to evaluate the CDF.
    
        The CDF is a step function jumping at the value of tensor, with shape (N, T, sp_size, X).
    """
    return (tensor.unsqueeze(-1) < xs).float()

    
def get_gmm_cdf(mixing: torch.Tensor, means: torch.Tensor, log_var: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    return GMMPredictionHead.get_mixture_cdf(mixing, means, log_var, xs)


def gmm_interval_coverage_and_width(
        tensors: List[torch.Tensor],
        tensor_names: List[str],
        xs: torch.Tensor,
        conf: float,
        sp_size: int = 20,
        gpu: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ This is a helper function to pass the function for confidence interval
        """
        # Define confidence interval function wrapper
        def ci_function(chunks, xs, conf):
            return GMMPredictionHead.get_confidence_interval(
                chunks["mixing"], chunks["means"], chunks["log_var"],
                xs, conf=conf
            )
        
        # Use the new generic function
        return interval_coverage_and_width(
            tensors=tensors,
            tensor_names=tensor_names,
            ci_function=ci_function,
            xs=xs,
            conf=conf,
            sp_size=sp_size,
            gpu=gpu
        )

def get_crps_gmm_vs_gt(mixing, means, log_var, xs, gt, sp_size=20, gpu=True):
    """
    Compute CRPS between GMM prediction and ground truth values.
    
    Args:
        mixing: GMM mixing coefficients (N, T, P, K)
        means: GMM means (N, T, P, K)
        log_var: GMM log variances (N, T, P, K)
        xs: Points to evaluate density (X,)
        gt: Ground truth data (N, T, P)
        vis: Whether to visualize GMM density
        sp_size: Chunk size for spatial dimension
        gpu: Whether to use GPU
        
    Returns:
        CRPS scores (N, T, P)
    """
    # Define CDF computation functions
    def gmm_cdf_func(chunks, xs):
        return get_gmm_cdf(chunks['mixing'], chunks['means'], chunks['log_var'], xs)
    
    def gt_cdf_func(chunks, xs):
        return get_point_cdf(chunks['gt'], xs)
    
    tensors = [mixing, means, log_var, gt]
    tensor_names = ["mixing", "means", "log_var", "gt"]
    scores = crps_from_cdf(tensors, tensor_names, gmm_cdf_func, gt_cdf_func, xs, sp_size, gpu)
    
    return scores


def ignore_score_when_gt_is(scores: torch.Tensor, gt: torch.Tensor, invalid_gt: float = float("nan"), ignore_value: float = float("nan")) -> torch.Tensor:
    """
    Ignore values in the scores tensor where the ground truth (gt) is NaN.
    This is needed because sometimes a gt is NaN, but its KNN is not entirely NaN.
    Without this function, we can still compute a CRPS score in theory, but it is not considered as valid.
    
    Args:
        scores: Tensor of CRPS scores
        gt: Ground truth tensor
    """
    if not np.isnan(invalid_gt):
        scores[gt == invalid_gt] = ignore_value
    
    # always ignore NaN values in gt
    scores[gt.isnan()] = ignore_value

    return scores
