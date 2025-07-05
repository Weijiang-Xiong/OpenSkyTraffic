import torch 
from einops import rearrange
from typing import List, Tuple

from .probabilistic import interval_coverage_and_width
from ...models.gmmpred import GMMPredictionHead

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

def get_knn_ecdf(inputs: torch.Tensor, gt: torch.Tensor, knn_nb: int, xs: torch.Tensor) -> torch.Tensor:
    gt_knn = get_knn_neighbors(inputs, gt, k=knn_nb)
    
    # after sorting, the NaN values are at the end of the tensor
    sorted_gt, _ = torch.sort(gt_knn, dim=-1)
    
    # reshape sorted_gt to (N, T, sp_size, k, 1), so the last dimension will be broadcasted with xs
    # knn_ecdf  = (sorted_gt.unsqueeze(-1) <= xs).sum(dim=-2).float() / knn_nb # → (N, T, sp_size, X)
    
    # since the KNN neighbors are sorted, we can use searchsorted to get the counts, i.e., how many 
    # neighbors are less than or equal to each value in xs (i.e., the index to insert xs in the sorted_gt)
    # this is more memory-friendly than the previous implementation, which tries to allocate (N, T, sp_size, k, X)
    N_dim, T_dim, sp_size_dim, _ = sorted_gt.shape
    counts = torch.searchsorted(
        # We need to replace the NaN values with inf because otherwise searchsorted will think the values should 
        # be inserted at the end of the tensor, and we always get k counts
        torch.nan_to_num(sorted_gt, nan=float("inf")), 
        xs.view(1,1,1,-1).expand(N_dim, T_dim, sp_size_dim, -1), 
        side='right')

    # we divide by the valid values (ignoring nan) to get the empirical CDF that always ends at 1.0
    num_valid_neighbors = torch.logical_not(gt_knn.isnan()).sum(dim=-1)
    knn_ecdf = counts.float() / num_valid_neighbors.unsqueeze(-1).float()
    
    # at this step, if an element in gt is NaN, the corresponding knn_ecdf will be NaN as well.
    # this will be addressed by self.invalid_to_ignore_value before returning the scores.
    return knn_ecdf

def get_knn_neighbors(x: torch.Tensor, y: torch.Tensor, k: int = 20) -> torch.Tensor:
    """
    For each spatial location, we compute the k nearest neighbors among all samples in the dataset (including self), according to the L2 distance in the input time series (drone speed, nan values replaced by mean). 
    The target output y is then gathered according to the k nearest neighbors, and they form an empirical distribution.
    
    Args:
        x: Data tensor of shape (N, T, P) to compute distances from
        y: Data tensor of shape (N, T, P) to gather neighbors from
        k: Number of nearest neighbors to find (default: 20)
        
    Returns:
        knn_y: Tensor of shape (N, T, P, k) containing the corresponding y values of k nearest neighbors
    """
    N, T, P = x.shape
    
    # Compute pairwise L2 distances between all N samples along T dimension for all P
    # Result shape: (N, N, P)
    dist = rearrange(
        torch.cdist(rearrange(x, "N T P -> P N T"), rearrange(x, "N T P -> P N T"), p=2.0),
        "P N1 N2 -> N1 N2 P",
    )
    # this is a previous implementation that is not memory efficient, because it tries to 
    # it will try to allocate (N, N, T, P) for the pairwise difference before taking the sum over T
    # dist = torch.sqrt(torch.sum((x.unsqueeze(1) - x.unsqueeze(0))**2, dim=2))

    # Find k nearest neighbors for each sample at each P location
    # Result shapes: (N, k, P)
    _, knn_indices = torch.topk(dist, k=k, dim=1, largest=False)

    # Gather the corresponding y values using advanced indexing
    # https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing
    # Result shape: (N, k, P, T)
    knn_y = y[knn_indices, # from the k nearest neighbors
                :, # keep all time steps 
                torch.arange(P).view(1, 1, P).expand(N, k, -1)]
    
    return rearrange(knn_y, 'N k P T -> N T P k')


def ignore_score_when_gt_isnan(scores: torch.Tensor, gt: torch.Tensor, ignore_value: float = float("nan")) -> torch.Tensor:
    """
    Ignore values in the scores tensor where the ground truth (gt) is NaN.
    This is needed because sometimes a gt is NaN, but its KNN is not entirely NaN.
    Without this function, we can still compute a CRPS score in theory, but it is not considered as valid.
    
    Args:
        scores: Tensor of CRPS scores
        gt: Ground truth tensor
    """
    scores[gt.isnan()] = ignore_value

    return scores