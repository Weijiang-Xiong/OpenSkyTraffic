import torch
import torch.nn as nn
import numpy as np
import logging
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple

from .metr_evaluation import MetrEvaluator
from .metrics import (
    gmm_interval_coverage_and_width,
    ignore_score_when_gt_is,
    get_crps_gmm_vs_gt,
)

logger = logging.getLogger("default")

class MetrGMMEvaluator(MetrEvaluator):
    """
    Evaluator for Gaussian Mixture Model predictions on METR dataset.
    Extends MetrEvaluator with uncertainty evaluation capabilities.
    """
    # confidence levels to evaluate
    eval_confs = np.round(np.arange(0.5, 1.0, 0.05), 2).tolist()  

    def __init__(self, 
                 save_dir: str = None, 
                 collect_pred=["pred", "mixing", "means", "log_var"], 
                 collect_data=["target"],
                 data_min: float = 0.0, 
                 data_max: float = 70.0,
                 sp_size: int = 5,
                 gpu: bool = True,
                 ci_pts: int = 500):
        super().__init__(save_dir, collect_pred, collect_data)
        self.data_min = data_min # minimum data value (traffic data is typically non-negative)
        self.data_max = data_max # maximum data value (adjustable based on dataset)
        self.sp_size = sp_size # chunk size for spatial dimension to save memory
        self.gpu = gpu # whether to use GPU acceleration for evaluation
        self.ci_pts = ci_pts # number of points to evaluate GMM density for confidence intervals

    def evaluate(self, model: nn.Module, dataloader: DataLoader, verbose: bool = False, visualize: bool = False) -> Dict[str, float]:
        """
        Evaluate GMM model with both deterministic and probabilistic metrics.
        
        Args:
            model: The GMM model to evaluate
            dataloader: Data loader for evaluation
            verbose: Whether to print detailed results
            
        Returns:
            Dictionary of evaluation metrics
        """
        # First get basic deterministic metrics from parent class
        _ = super().evaluate(model, dataloader, verbose=verbose, visualize=visualize)
        
        # Collect predictions and data for GMM evaluation
        all_preds, all_data = self.collect_predictions(
            model, dataloader, 
            pred_seqs=self.collect_pred, 
            data_seqs=self.collect_data
        )
        
        if verbose:
            logger.info("Evaluating CRPS scores")
        self.evaluate_crps(all_preds, all_data, verbose=verbose)
        
        if verbose:
            logger.info("Evaluating confidence intervals")
        self.evaluate_confidence_interval(all_preds, all_data, verbose=verbose)
        
        # Update results with GMM metrics
        return self.metrics_scalar

    def evaluate_crps(self, all_preds: Dict[str, torch.Tensor], all_data: Dict[str, torch.Tensor], verbose: bool = False):
        """
        Evaluate CRPS (Continuous Ranked Probability Score) between GMM predictions and ground truth.
        
        Args:
            all_preds: Dictionary containing model predictions
            all_data: Dictionary containing ground truth data
            verbose: Whether to print results
        """
        # Define evaluation points
        xs = torch.linspace(self.data_min, self.data_max, self.ci_pts)
        
        # Evaluate CRPS between GMM and ground truth
        crps_scores = get_crps_gmm_vs_gt(
            mixing=all_preds["mixing"],
            means=all_preds["means"],
            log_var=all_preds["log_var"],
            xs=xs,
            gt=all_data["target"],
            sp_size=self.sp_size,
            gpu=self.gpu
        )
        
        # Ignore scores where ground truth is NaN
        crps_scores = ignore_score_when_gt_is(crps_scores, all_data["target"], invalid_gt=0.0)
        
        # Analyze and store scores
        self.analyze_scores(crps_scores, note="CRPS_GMM_GT", verbose=verbose)

    def evaluate_confidence_interval(self, all_preds: Dict[str, torch.Tensor], all_data: Dict[str, torch.Tensor], verbose: bool = False):
        """
        Evaluate confidence interval coverage and width for different confidence levels.
        
        Args:
            all_preds: Dictionary containing model predictions
            all_data: Dictionary containing ground truth data
            verbose: Whether to print results
        """
        xs = torch.linspace(self.data_min, self.data_max, self.ci_pts)
        tensors = [all_preds["mixing"], all_preds["means"], all_preds["log_var"], all_data["target"]]
        tensor_names = ["mixing", "means", "log_var", "gt"]
        
        for conf in self.eval_confs:
            within_ci, interval_width = gmm_interval_coverage_and_width(
                tensors=tensors,
                tensor_names=tensor_names,
                xs=xs,
                conf=conf,
                sp_size=self.sp_size,
                gpu=self.gpu
            )
            
            # Ignore scores where ground truth is NaN
            within_ci = ignore_score_when_gt_is(within_ci, all_data["target"], invalid_gt=0.0)
            interval_width = ignore_score_when_gt_is(interval_width, all_data["target"], invalid_gt=0.0)
            
            # Store individual confidence level results
            self.analyze_scores(within_ci.float(), note=f"CI_COVER_{conf}", verbose=False)
            self.analyze_scores(interval_width, note=f"CI_WIDTH_{conf}", verbose=False)
        
        # Compute overall calibration metrics
        self.compute_calibration_metrics(verbose=verbose)

    def compute_calibration_metrics(self, verbose: bool = False):
        """
        Compute overall calibration metrics from individual confidence level results.
        
        Args:
            verbose: Whether to print results
        """
        CCE_conf_horizon, AW_conf_horizon = [], []
        
        for conf in self.eval_confs:
            # Confidence Calibration Error: difference between confidence level and actual coverage
            CCE_conf_horizon.append([abs(x - conf) for x in self.metrics_vector[f'CI_COVER_{conf}']])
            AW_conf_horizon.append(self.metrics_vector[f'CI_WIDTH_{conf}'])
        
        # Average over confidence levels
        mCCE_horizon = np.stack(CCE_conf_horizon, axis=-1).mean(axis=-1)
        mAW_horizon = np.stack(AW_conf_horizon, axis=-1).mean(axis=-1)
        
        # Store overall metrics
        self.metrics_scalar['mCCE'] = mCCE_horizon.mean().item()
        self.metrics_scalar['mAW'] = mAW_horizon.mean().item()
        self.metrics_vector['mCCE'] = mCCE_horizon.tolist()
        self.metrics_vector['mAW'] = mAW_horizon.tolist()
        
        if verbose:
            logger.info(f"Mean Confidence Calibration Error (mCCE): {self.metrics_scalar['mCCE']:.4f}")
            logger.info(f"Mean Average Width (mAW): {self.metrics_scalar['mAW']:.4f}")
