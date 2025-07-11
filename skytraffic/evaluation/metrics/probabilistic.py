import torch
import logging
import numpy as np
from typing import Dict, Tuple, List, Callable

def gaussian_dist_metrics(pred: torch.Tensor, target: torch.Tensor, scale:torch.Tensor, 
                        offset_coeffs: Dict[float, float], ignore_value=0.0, verbose=False):
    """ The uncertainty prediction is evaluated by the corresponding confidence intervals, where we assume the interval is centered on the expected value, and the upper bound and lower bound 
    are expressed by multiples of the predicted uncertainty scale. 
    
    The primary metrics are 
        1. mAO, the half-width of the confidence intervals averaged over all confidences and all predictions. 
        2. mCP, the percentage of data points covered by the confidence interval, averaged over all confidences 
        3. mCCE, the difference between confidence and data coverage, averaged over all confidences
    
    A calibrated model is expected to predict confidence intervals whose coverage is the same as the confidence score. That will result in a zero mCCE. 
    
    Args:
        pred (torch.Tensor): the expected future value predicted by the model
        target (torch.Tensor): the real future value from data
        scale (torch.Tensor): the predicted uncertainty scale, have the same unit as `pred`
        offset_coeffs (Dict[float, float]): pairs of confidence score and interval offsets
        ignore_value (float, optional): the values corresponding to "no data". Defaults to 0.0.
        verbose (bool, optional): whether to print evaluation results. Defaults to False.

    Returns:
        res: collection of evaluation results
    """
    pred, target, scale = pred.flatten(), target.flatten(), scale.flatten()
    if ignore_value is not None:
        valid = (target != ignore_value)
        pred, target, scale = pred[valid], target[valid], scale[valid]
    
    Conf, OC_val, CP, CCE, AO = [[] for _ in range(5)]
    for confidence, offset in offset_coeffs.items():
        ub = pred + offset * scale
        lb = pred - offset * scale
        
        covered = torch.logical_and(target < ub, target > lb)
        coverage_percentage = covered.sum() / covered.numel()
        
        Conf.append(confidence)
        OC_val.append(offset)
        CP.append(coverage_percentage.item())
        CCE.append(abs((confidence - coverage_percentage).item()))
        AO.append(torch.mean(offset*scale).item())

    res = { 
           "mAO"                : sum(AO)/len(AO), # mean average offset, interval width
           "mCP"                : sum(CP)/len(CP), # mean coverage percentage 
           "mCCE"               : sum(CCE)/len(CCE), # mean confidence calibration error 
           "eval_points"        : Conf,
           "offset_coeffs"      : OC_val,
           "coverage_percentage": CP,
           "average_offset"     : AO,
           "calibration_error"  : CCE
    }
    
    if verbose:
        logger = logging.getLogger("default")
        logger.info("Uncertainty Metrics")
        logger.info(" ".join(["{}: {:.3f},".format(k, v) for k, v in res.items() if isinstance(v, (int, float))]))
        logger.info("Evaluated confidence interval {}".format(Conf))
        logger.info("Corresponding data coverage percentage {} \n".format(np.round(CP, 2).tolist()))

    return res


def crps_from_cdf(
        tensors: List[torch.Tensor], 
        tensor_names: List[str], 
        cdf_func1: Callable, 
        cdf_func2: Callable, 
        xs: torch.Tensor, 
        sp_size: int = 20, 
        gpu: bool = True
    ) -> torch.Tensor:
    """
    Compute CRPS between two distributions.

    Computing the CRPS requires the CDFs, which we obtain by a cumsum of the PDF. 
    Storing the GMM density for the whole dataset will cost N * T * P * X * 4 Byte.
    For the Simbarca test set and density evaluated at 1000 points, that means 30 GB. 
    If we store the density per GMM component, the cost will be multiplied by the number of components, e.g., K=5. 
    Looping over the spatial locations and doing evaluation separately for them is correct but not efficient.
    So here we implement a batch-wise evaluation, where we split the tensors along the spatial location dimension to get chunks with size sp_size.
    
    Args:
        tensors: List of tensor inputs needed for composing the CDFs
        tensor_names: List of names corresponding to the tensors
        cdf_func1: First CDF computation function (prediction)
        cdf_func2: Second CDF computation function (reference)
        xs: Points to evaluate CDFs at (also probability density), shape (X,)
        sp_size: Chunk size for spatial dimension splitting
        gpu: Whether to use GPU acceleration
        
    Returns:
        CRPS scores with shape (N, T, P)
    """
    
    # Split tensors along spatial dimension to save memory
    split_tensors = {}
    for name, tensor in zip(tensor_names, tensors):
        split_tensors[name] = torch.split(tensor, sp_size, dim=2)
    
    CRPS_by_chunk = []
    # Get the number of chunks from any tensor (they all have the same number)
    num_chunks = len(split_tensors[tensor_names[0]])
    
    for i in range(num_chunks):
        # Extract current chunk for each tensor
        chunks = {name: split_tensors[name][i] for name in tensor_names}

        if gpu and torch.cuda.is_available():
            chunks = {k: v.cuda() for k, v in chunks.items()}
            xs = xs.to('cuda')
        
        # Compute CDFs using the provided functions
        cdf1 = cdf_func1(chunks, xs)
        cdf2 = cdf_func2(chunks, xs)
        
        # Compute CRPS
        CRPS_chunk = torch.sum((cdf1 - cdf2)**2 * abs(xs[1] - xs[0]), dim=-1)
        CRPS_by_chunk.append(CRPS_chunk)
    
    # Concatenate the chunks
    CRPS = torch.cat(CRPS_by_chunk, dim=-1).cpu()
    
    return CRPS

def interval_coverage_and_width(
    tensors: List[torch.Tensor],
    tensor_names: List[str],
    ci_function: Callable,
    xs: torch.Tensor,
    conf: float,
    sp_size: int = 20,
    gpu: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        For each point in gt, compute whether it is covered by the `conf` confidence interval specified by the `ci_function`,
        as well as the width of the predicted confidence interval.
        
        Args:
            tensors: List of tensor inputs needed for composing the CDFs
            tensor_names: List of names corresponding to the tensors
            ci_function: Function to compute the confidence interval
            xs: Points to evaluate CDFs at (also probability density), shape (X,)
            conf: the confidence level
            sp_size: Chunk size for spatial dimension splitting
            gpu: Whether to use GPU acceleration

        Returns:
            within_ci: a boolean tensor indicating whether the ground truth value is within the confidence interval, shape (N, T, P)
            interval_width: the average width of the confidence interval, shape (N, T, P)
    """

    # split tensors along spatial dimension to save memory
    split_tensors = {}
    for name, tensor in zip(tensor_names, tensors):
        split_tensors[name] = torch.split(tensor, sp_size, dim=2)
    

    score_by_chunk = {"within_ci": [], "interval_width": []}
    # Get the number of chunks from any tensor (they all have the same number)
    num_chunks = len(split_tensors[tensor_names[0]])
    for i in range(num_chunks):
        # Extract current chunk for each tensor
        chunks = {name: split_tensors[name][i] for name in tensor_names}

        # put the tensors on GPU if available
        if gpu and torch.cuda.is_available():
            chunks = {name: tensor.cuda() for name, tensor in chunks.items()}

        lb, ub = ci_function(chunks, xs, conf)
        
        # Compute the percentage of predictions within the confidence interval
        # a confidence interval can contain at most K subintervals, where K is the number of GMM components
        # if any subinterval covers the ground truth value, the prediction is within the confidence interval
        within_ci = torch.any(
            torch.logical_and( 
                lb <= chunks["gt"].unsqueeze(-1), 
                ub >= chunks["gt"].unsqueeze(-1)
            ), 
            dim=-1
        ) # (N, T, P)
        interval_width = torch.abs(ub - lb).sum(dim=-1) # (N, T, P)
        score_by_chunk['within_ci'].append(within_ci)
        score_by_chunk['interval_width'].append(interval_width)

    # Concatenate the chunks
    within_ci = torch.cat(score_by_chunk['within_ci'], dim=-1)
    interval_width = torch.cat(score_by_chunk['interval_width'], dim=-1)

    return within_ci, interval_width