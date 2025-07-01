import json
import logging
import numpy as np 
from typing import Dict, List, Tuple

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns 

from einops import rearrange

from ..utils.io import make_dir_if_not_exist
from ..models.gmmpred import GMMPredictionHead
from ..data.datasets import SimBarcaMSMT
from .simbarca_evaluation import SimBarcaEvaluator

sns.set_style('darkgrid')
logger = logging.getLogger("default")

class SimBarcaGMMEvaluator(SimBarcaEvaluator):
    """ Notation on tensor shapes.

        N: number of samples
        T: number of time steps
        P: number of spatial locations
        C: number of features (value and time)
        K: number of GMM components
        X: number of points to evaluate density

        Note that NaN values can exist in the input, groundtruth and scores, so whenever we do an average, we 
        need to use torch.nanmean instead of a simple mean.
    """
    # these are the additional output sequences other than the predicted targets
    # for GMM prediction, that would be the GMM parameters for segments and regions
    add_output_seq = ["seg_mixing","seg_means","seg_log_var","reg_mixing","reg_means","reg_log_var"]
    eval_tasks = ["seg", "reg"]
    data_min = {"seg": 0.0, "reg": 0.0}
    data_max = {"seg": 14.0, "reg": 9.0}
    seq_labels_by_task = {"seg": "pred_speed", "reg": "pred_speed_regional"}
    # the confidence levels to evaluate, from 0.5 to 0.95 with step 0.05
    eval_confs = np.round(np.arange(0.5, 1.0, 0.05), 2).tolist()
    ci_pts = 500 # the number of points to evaluate the GMM density, for confidence interval
    vis_pts = 300 # the number of points to visualize the GMM density, for visualization
    density_scale = 5 # divide by this to scale the GMM density values in visualization
    input_window = 30 # the input window size in minutes, default is 30 min
    data_time_step = 3 # the input step size in minutes, default is 3 min
    
    def __init__(
        self,
        ignore_value=float("nan"),
        mape_threshold=1.0,
        save_dir: str = None,
        save_note="",
        visualize=False,
        add_output_seq: list = None,
    ) -> None:
        super().__init__(ignore_value, mape_threshold, save_dir, save_note, visualize)
        if add_output_seq is not None:  # overwrite the default if provided
            self.add_output_seq = add_output_seq
        self.sp_size = 20  # the size of the chunks to split the tensors space dimension, default 10
        self.knn_nb = 20  # the number of nearest neighbors to find, default 20
        self.gpu = True  # whether to use GPU acceleration, default True

    
    #########################################################################################
    ################ Functions for collecting data and statistics         ###################
    #########################################################################################
    def collect_predictions(self, model, data_loader, pred_seqs = None, data_seqs = None):
        
        all_preds, all_data = super().collect_predictions(
            model=model, data_loader=data_loader, 
            pred_seqs=pred_seqs, data_seqs=data_seqs
        )
        if "drone_speed" in all_data:
            all_data["drone_speed"] = all_data["drone_speed"][..., 0].nan_to_num(
            nan=data_loader.dataset.metadata['mean_and_std']['drone_speed'][0]
            )
            
        return all_preds, all_data
    
    
    def analyze_scores(self, scores, note:str=None, verbose=False):
        """ Given an error score tensor of shape (N, T, P), we summarize the score at each time step.
            The summary is saved to self.saved_metrics, with the key being the note.
            Note that the scores need to be filtered with self.ignore_score_when_gt_isnan, otherwise the aggregation will be incorrect. 
            The reason for not putting self.ignore_score_when_gt_isnan in the analyze_scores function is that we may want to use other analysis procedure, rather than just averaging for each time step. So we leave it to the caller to do the filtering.
            
            Args:
                scores: Error scores with shape (N, T, P)
                note: Key to use when saving to self.saved_metrics
                verbose: Whether to print the summary
        """
        if note is None or note == "":
            logger.error(f"The note can not be {note}, please use a different one.")
        
        # Save average score at each time step (averaging over samples N and spatial locations P)
        self.saved_scores['vector'][note] = torch.nanmean(scores, dim=(0, 2)).cpu().numpy()
        # Save the mean over all the time steps, ignoring NaN values
        self.saved_scores['scalar'][f'{note}'] = torch.nanmean(scores).item()
        
        if verbose:
            logger.info(f"Saved average {note} scores by time step to saved_metrics")
            logger.info(f"Overall average {note}: {self.saved_scores['scalar'][f'{note}']:.4f}")


    def ignore_score_when_gt_isnan(self, scores, gt) -> torch.Tensor:
        """
        Ignore values in the scores tensor where the ground truth (gt) is NaN.
        This is needed because sometimes a gt is NaN, but its KNN is not entirely NaN.
        Without this function, we can still compute a CRPS score in theory, but it is not considered as valid.
        
        Args:
            scores: Tensor of CRPS scores
            gt: Ground truth tensor
        """
        scores[gt.isnan()] = self.ignore_value

        return scores
    

    #########################################################################################
    ################ Main evaluation routines           #####################################
    #########################################################################################
    def evaluate(self, model: nn.Module, data_loader: DataLoader, verbose=False) -> Dict[str, float]:
        """ This evaluation function is structured as follows:
                0. run the super class's evaluation function (deterministic metrics)
                1. collect predictions and required data sequences from the model and dataset
                2. run various evaluation functions (CRPS, CI, etc.), average scores saved to self.saved_scores
                3. if visualization is required, plot the scores and predictions
                4. return the average scores (so that the engine and hooks can have access to the main scores at this round)
        """
        _ = super().evaluate(model, data_loader, verbose=verbose)
        
        dataset: SimBarcaMSMT = data_loader.dataset
        soi: List[int] = dataset.sections_of_interest
        s2i = dataset.section_id_to_index  # section ID to array index in space dimension
        session_ids = dataset.session_ids  # list of simulation session IDs

        all_preds, all_data = self.collect_predictions(
            model,
            data_loader,
            pred_seqs=dataset.output_seqs + self.add_output_seq,
            data_seqs=dataset.output_seqs + ['drone_speed'],
        )
        logger.info("Evaluating CRPS scores")
        self.evaluate_crps(all_preds, all_data, verbose=verbose)
        logger.info("Evaluating confidence intervals")
        self.evaluate_confidence_interval(all_preds, all_data, verbose=verbose)
        
        if self.visualize:
            
            self.plot_crps_scores()
            self.save_scores_to_json()
            
            # Pass lists of positions and sessions instead of looping here
            p_list_seg = [s2i[sec] for sec in soi]
            s_list = list(range(len(session_ids)))
            p_list_reg = list(range(len(dataset.cluster_id.unique()))) # For regional predictions
            
            fix_time_pred_path = f"{self.save_dir}/fix_time_pred"
            make_dir_if_not_exist(fix_time_pred_path)
            logger.info(f"Plotting predictions at a fixed timestamp, files saved to {fix_time_pred_path}")

            self.plot_pred_fix_time(all_preds, all_data,
                    p_list=p_list_seg, s_list=s_list,
                    time_step_to_viz=10, pred_horizons=10,
                    sample_per_session=dataset.sample_per_session, task="seg",
                    sec_ids=soi, sim_ids=session_ids, subfolder_path=fix_time_pred_path)
            
            gmm_pred_path = f"{self.save_dir}/gmm_predictions"
            make_dir_if_not_exist(gmm_pred_path)
            logger.info(f"Plotting GMM predictions, files saved to {gmm_pred_path}")

            self.plot_30min_gmm_preds(all_preds, all_data, p_list=p_list_seg, s_list=s_list, 
                    sample_per_session=dataset.sample_per_session, task="seg", sim_ids=session_ids, 
                    sec_ids=soi, with_knn=True, subfolder_path=gmm_pred_path)
            self.plot_30min_gmm_preds(all_preds, all_data, p_list=p_list_reg, s_list=s_list,
                    sample_per_session=dataset.sample_per_session, task="reg", sim_ids=session_ids,
                    sec_ids=p_list_reg, subfolder_path=gmm_pred_path)
    
        return self.saved_scores['scalar']


    #########################################################################################
    ################ Evaluation subroutines        ##########################################
    #########################################################################################
    def evaluate_confidence_interval(self, all_preds, all_data, verbose=False):
        """
        This function evaluates the confidence interval of the GMM prediction.
        """
        for conf in self.eval_confs:
            within_ci, interval_width = self.point_wise_cover_and_width(
                mixing=all_preds["seg_mixing"],
                means=all_preds["seg_means"],
                log_var=all_preds["seg_log_var"],
                gt=all_data["pred_speed"],
                xmin=self.data_min["seg"],
                xmax=self.data_max["seg"],
                n_points=self.ci_pts,
                conf=conf,
                sp_size=self.sp_size,
                gpu=self.gpu,
            )

            within_ci = self.ignore_score_when_gt_isnan(within_ci, all_data["pred_speed"])
            interval_width = self.ignore_score_when_gt_isnan(interval_width, all_data["pred_speed"])

            self.analyze_scores(
                scores=within_ci.float(),
                note=f"CI_COVER_{conf}",
                verbose=False, # don't separately print the scores for each confidence level
            )
            self.analyze_scores(
                scores=interval_width,
                note=f"CI_WIDTH_{conf}",
                verbose=False, # don't separately print the scores for each confidence level
            )
        
        # from the saved CI_COVER and CI_WIDTH, compute the calibration error and average width
        # confidence calibration error is the absolute difference between confidence level and the average coverage
        CCE_conf_horizon, AW_conf_horizon = [], []
        for conf in self.eval_confs:
            CCE_conf_horizon.append(abs(self.saved_scores['vector'][f'CI_COVER_{conf}'] - conf))
            AW_conf_horizon.append(self.saved_scores['vector'][f'CI_WIDTH_{conf}'])

        # concatenate the scores along a new dimension, which is the confidence level
        CCE_horizon = np.stack(CCE_conf_horizon, axis=-1).mean(axis=-1)
        AW_horizon = np.stack(AW_conf_horizon, axis=-1).mean(axis=-1)

        # save the scores 
        self.saved_scores['scalar']['mCCE'] = CCE_horizon.mean().item()
        self.saved_scores['scalar']['mAW'] = AW_horizon.mean().item()
        self.saved_scores['vector']['CCE_horizon'] = CCE_horizon.tolist()
        self.saved_scores['vector']['AW_horizon'] = AW_horizon.tolist()
        if verbose:
            logger.info(f"mCCE: {self.saved_scores['scalar']['mCCE']:.4f}, mAW: {self.saved_scores['scalar']['mAW']:.4f}")


    def evaluate_crps(self, all_preds, all_data, verbose=False):

        cdf_xs = torch.linspace(self.data_min['seg'], self.data_max['seg'] , self.ci_pts)

        self.analyze_scores(
            scores=self.ignore_score_when_gt_isnan(
                self.get_crps_gmm_vs_emp_dist(
                    mixing=all_preds["seg_mixing"],
                    means=all_preds["seg_means"],
                    log_var=all_preds["seg_log_var"],
                    xs=cdf_xs,
                    inputs=all_data["drone_speed"],
                    gt=all_data["pred_speed"],
                    sp_size=self.sp_size,
                    knn_nb=self.knn_nb,
                    gpu=self.gpu,
                ),
                all_data["pred_speed"],
            ),
            note="CRPS_GMM_EMP",
            verbose=verbose,
        )

        self.analyze_scores(
            scores=self.ignore_score_when_gt_isnan(
                self.get_crps_pred_vs_emp_dist(
                    pred=all_preds["pred_speed"],
                    xs=cdf_xs,
                    inputs=all_data["drone_speed"],
                    gt=all_data["pred_speed"],
                    sp_size=self.sp_size,
                    knn_nb=self.knn_nb,
                    gpu=self.gpu,
                ),
                all_data["pred_speed"],
            ),
            note="CRPS_PRED_EMP",
            verbose=verbose,
        )

        self.analyze_scores(
            scores=self.ignore_score_when_gt_isnan(
                self.get_crps_gmm_vs_gt(
                    mixing=all_preds["seg_mixing"],
                    means=all_preds["seg_means"],
                    log_var=all_preds["seg_log_var"],
                    xs=cdf_xs,
                    gt=all_data["pred_speed"],
                    sp_size=self.sp_size,
                    gpu=self.gpu,
                ),
                all_data["pred_speed"],
            ),
            note="CRPS_GMM_GT",
            verbose=verbose,
        )


    #########################################################################################
    ################ Functions utilized in the subroutines        ###########################
    #########################################################################################
    @staticmethod
    def point_wise_cover_and_width(
        mixing: torch.Tensor,
        means: torch.Tensor,
        log_var: torch.Tensor,
        gt: torch.Tensor,
        xmin: float,
        xmax: float,
        n_points: int,
        conf: float,
        sp_size: int = 20,
        gpu: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            For each point in gt, compute whether it is covered by the `conf` confidence interval of the GMM prediction,
            as well as the width of the confidence interval.
            
            Args:
                mixing: the GMM mixing coefficients, shape (N, T, P, K)
                means: the GMM means, shape (N, T, P, K)
                log_var: the GMM variances, shape (N, T, P, K)
                gt: the ground truth values, shape (N, T, P)
                xmin: the minimum value of the predicted variable
                xmax: the maximum value of the predicted variable
                n_points: the number of points to evaluate the GMM density
                conf: the confidence level

            Returns:
                within_ci: a boolean tensor indicating whether the ground truth value is within the confidence interval, shape (N, T, P)
                interval_width: the average width of the confidence interval, shape (N, T, P)
        """

        # split tensors along spatial dimension to save memory
        tensor_names = ["mixing", "means", "log_var", "gt"]
        tensors = [mixing, means, log_var, gt]
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

            gmm_confidence_intervals = GMMPredictionHead.get_confidence_interval(
                chunks["mixing"], chunks["means"], chunks["log_var"],
                xmin=xmin, xmax=xmax, n_points=n_points, conf=conf
            )
            
            # Compute the percentage of predictions within the confidence interval
            # a confidence interval can contain at most K subintervals, where K is the number of GMM components
            # if any subinterval covers the ground truth value, the prediction is within the confidence interval
            within_ci = torch.any(
                torch.logical_and( 
                    gmm_confidence_intervals[0] <= chunks["gt"].unsqueeze(-1), 
                    gmm_confidence_intervals[1] >= chunks["gt"].unsqueeze(-1)
                ), 
                dim=-1
            ) # (N, T, P)
            interval_width = torch.abs(gmm_confidence_intervals[1] - gmm_confidence_intervals[0]).sum(dim=-1) # (N, T, P)
            score_by_chunk['within_ci'].append(within_ci)
            score_by_chunk['interval_width'].append(interval_width)
    
        # Concatenate the chunks
        within_ci = torch.cat(score_by_chunk['within_ci'], dim=-1)
        interval_width = torch.cat(score_by_chunk['interval_width'], dim=-1)

        return within_ci, interval_width


    @staticmethod
    def _compute_crps(tensors, tensor_names, cdf_func1, cdf_func2, xs, sp_size=20, gpu=True):
        """
        Compute CRPS between two distributions.
        
        Args:
            tensors: List of tensor inputs needed for composing the CDFs
            tensor_names: List of names corresponding to the tensors
            cdf_func1: First CDF computation function (prediction)
            cdf_func2: Second CDF computation function (reference)
            xs: Points to evaluate CDFs at, shape (X,)
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

    
    def get_crps_gmm_vs_emp_dist(self, mixing, means, log_var, xs, inputs, gt, sp_size=20, knn_nb=20, gpu=True):
        """
        This function computes the CRPS score between the GMM prediction and an empirical distribution.
        The empirical distribution of a sample is computed by taking the ground truth of a set of samples, 
        whose inputs are the K nearest neighbors of the inputs in this sample.  
        
        Storing the GMM density for the whole dataset will cost N * T * P * X * 4 Byte.
        For the Simbarca test set and density evaluated at 1000 points, that means 30 GB. 
        If we store the density per GMM component, the cost will be multiplied by the number of components, e.g., K=5. 
        Looping over the spatial locations and doing evaluation separately for them is correct but not efficient.
        So here we implement a batch-wise evaluation, where we split the tensors along the spatial location dimension to get chunks with size sp_size.
        
        Args:
            mixing: the GMM mixing coefficients, with shape (N, T, P, K)
            means: the GMM means, with shape (N, T, P, K)
            log_var: the GMM variances, with shape (N, T, P, K)
            xs: the points to evaluate the GMM density, with shape (X,)
            inputs: the input data (value only, no time step), with shape (N, T, P)
            gt: the ground truth data, with shape (N, T, P)
            sp_size: the size of the chunks to split the tensors, default is 50
            knn_nb: the number of nearest neighbors to find, default is 20
            vis: whether to visualize the GMM density, default is False
            
        Returns:
            CRPS_emp: the CRPS score between predicted and empirical distribution, with shape (N, T, P)
        """
        
        def gmm_cdf_func(chunks, xs):
            return self.get_gmm_cdf(chunks["mixing"], chunks["means"], chunks["log_var"], xs)
        def knn_ecdf_func(chunks, xs):
            return self.get_knn_ecdf(chunks["inputs"], chunks["gt"], knn_nb, xs)
        
        # split the tensors along the spatial location dimension to get chunks, save memory
        tensors = [mixing, means, log_var, inputs, gt]
        tensor_names = ["mixing", "means", "log_var", "inputs", "gt"]
        scores = self._compute_crps(tensors, tensor_names, gmm_cdf_func, knn_ecdf_func, xs, sp_size, gpu)
        
        return scores
    
    
    def get_crps_gmm_vs_gt(self, mixing, means, log_var, xs, gt, sp_size=20, gpu=True):
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
            return self.get_gmm_cdf(chunks['mixing'], chunks['means'], chunks['log_var'], xs)
        
        def gt_cdf_func(chunks, xs):
            return self.get_point_cdf(chunks['gt'], xs)
        
        tensors = [mixing, means, log_var, gt]
        tensor_names = ["mixing", "means", "log_var", "gt"]
        scores = self._compute_crps(tensors, tensor_names, gmm_cdf_func, gt_cdf_func, xs, sp_size, gpu)
        
        return scores


    def get_crps_pred_vs_emp_dist(self, pred, xs, inputs, gt, sp_size=20, knn_nb=20, gpu=True):
        """
        Compute CRPS between point prediction and empirical distribution.
        
        Args:
            pred: Point predictions (N, T, P)
            xs: Points to evaluate density (X,)
            inputs: Input data (N, T, P)
            gt: Ground truth data (N, T, P)
            sp_size: Chunk size for spatial dimension
            knn_nb: Number of nearest neighbors
            vis: Whether to visualize prediction
            gpu: Whether to use GPU
            
        Returns:
            CRPS scores (N, T, P)
        """
        # Define CDF computation functions
        def pred_cdf_func(chunks, xs):
            return self.get_point_cdf(chunks['pred'], xs)
        
        def knn_cdf_func(chunks, xs):
            return self.get_knn_ecdf(chunks['inputs'], chunks['gt'], knn_nb, xs)
        
        tensors = [pred, inputs, gt]
        tensor_names = ["pred", "inputs", "gt"]
        scores = self._compute_crps(tensors, tensor_names, pred_cdf_func, knn_cdf_func, xs, sp_size, gpu)
        
        return scores


    def get_point_cdf(self, tensor, xs):
        """ tensor has shape (N, T, sp_size), can be ground-truth or predicted most-likely values.
            xs has shape (X,), which is the points to evaluate the CDF.
        
            The CDF is a step function jumping at the value of tensor, with shape (N, T, sp_size, X).
        """
        point_cdf = (tensor.unsqueeze(-1) < xs).float()
        
        return point_cdf

        
    def get_gmm_cdf(self, mixing, means, log_var, xs):
        gmm_density = GMMPredictionHead.get_mixture_density(mixing, means, log_var, xs)
        dx = abs(xs[1] - xs[0]) # the step size of the x values
        gmm_cdf = torch.cumsum(gmm_density * dx, axis=-1)
        gmm_cdf = gmm_cdf / gmm_cdf[..., -1].unsqueeze(-1) # ensure the CDF ends at 1
        
        return gmm_cdf
    
    
    def get_knn_ecdf(self, inputs, gt, knn_nb, xs):
        gt_knn = self.get_knn_neighbors(inputs, gt, k=knn_nb)
        
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
    
    @staticmethod
    def get_knn_neighbors(x, y, k=20):
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
    
    #########################################################################################
    ############### Functions for plotting statistics        ################################
    #########################################################################################

    def plot_crps_scores(self):
        """ 
            Plot the scores saved in self.saved_scores, which is a dictionary with keys being the score types.
            The values are numpy arrays with shape (T,) or (P,) for each time step and spatial location.
        """
        scores_by_time = self.saved_scores["vector"]
        # plot the scores by time step
        plot_legends_emp = [
            ("CRPS_GMM_EMP", "GMM vs Empirical"),
            ("CRPS_PRED_EMP", "Pred vs Empirical")
        ]
        plot_legends_gt = [            
            ("CRPS_GMM_GT", "GMM vs GT"),
            ("pred_speed_mae", "Pred vs GT (MAE)")
        ]
        for plot_legends, base_name in zip([plot_legends_emp, plot_legends_gt], ["emp", "gt"]):
            fig, ax = plt.subplots(figsize=(6, 5))
            for seq_name, seq_legend in plot_legends:
                if seq_name in scores_by_time.keys():
                    v = scores_by_time[seq_name]
                # plot the scores by time step using the specified line style
                ax.plot(v, label=seq_legend)
                
            ax.set_xticks(np.arange(len(v)))
            ax.set_xticklabels(3 * (np.arange(len(v))+1) )
            ax.set_xlabel("Prediction Time Horizion (min)")
            ax.set_ylabel("Score")
            ax.legend(loc="upper left")
            ax.set_title("CRPS by Time Horizion")
            plt.tight_layout()
            save_path = f"{self.save_dir}/crps_by_time_{base_name}_{self.save_note}.pdf"
            plt.savefig(save_path)
            logger.info(f"Save scores by time step to {save_path}")
            plt.close(fig)  # Close figure to free memory

    
    def save_scores_to_json(self, file_name: str = "final_evaluation_scores.json"):
        """
        Save the scores to a JSON file.
        The scores are saved in a dictionary with keys being the score types and values being the scores.
        """

        scalar_res = {k:float(v) for k, v in self.saved_scores['scalar'].items()}
        vector_res = {k:v for k, v in self.saved_scores['vector'].items() if isinstance(v, list)}
        res_to_save = {
            "average": scalar_res,
            "horizon": vector_res
        }

        save_path = f"{self.save_dir}/{file_name}"
        with open(save_path, 'w') as f:
            json.dump(res_to_save, f, indent=4)
        logger.info(f"Saved scores to {save_path}")

    #########################################################################################
    ############### Functions for plotting time-series predictions        ###################
    #########################################################################################

    def plot_30min_gmm_preds(
        self,
        all_preds,
        all_data,
        p_list,
        s_list,
        sample_per_session=20,
        task=None,
        sec_ids: List = None,
        sim_ids: List = None,
        with_knn=False,
        verbose=False,
        subfolder_path=None,
    ):
        """ 
            Plot the GMM prediction for positions `p_list` at simulation sessions `s_list`
            all_preds: dictionary of all predictions, values are tensors of shape (N, T, P, K) for GMM parameters and (N, T, P) for predicted speed
            all_data: dictionary of all data sequences (input and output alike), values are tensors of shape (N, T_i, P) where T_i may vary
            p_list: list of segment indices
            s_list: list of session indices 
            sample_per_session: sample per session, 20 for simbarca
            task: 'seg' or 'reg' to specify which task to plot (overrides regional flag)
            regional: whether to plot regional (True) or segment (False) predictions
            sec_ids: list of section IDs in aimsun, will be used in file name
            sim_ids: list of simulation session IDs, will be used in file name
            with_knn: whether to plot KNN empirical distribution, can be used with segment level predictions
        """
        # Determine which task to use
        assert task in self.eval_tasks, f"Task should be one of {self.eval_tasks}"
        
        ymin = self.data_min[task]
        ymax = self.data_max[task]
        label_seq = self.seq_labels_by_task[task]
        
        y_vals = torch.linspace(ymin, ymax, self.vis_pts)
        num_sessions = len(all_preds[label_seq]) // sample_per_session
        
        # Put everything to numpy once
        all_preds = {k: v for k, v in all_preds.items()}
        all_data = {k: v for k, v in all_data.items()}
        
        # Split arrays by session for efficiency (do this once), take last time step prediction (30 min)
        pred_by_session = torch.tensor_split(all_preds[label_seq][:, -1], num_sessions)
        gt_by_session = torch.tensor_split(all_data[label_seq][:, -1], num_sessions)
        mixing_by_session = torch.tensor_split(all_preds[f"{task}_mixing"][:, -1], num_sessions)
        means_by_session = torch.tensor_split(all_preds[f"{task}_means"][:, -1], num_sessions)
        logvar_by_session = torch.tensor_split(all_preds[f"{task}_log_var"][:, -1], num_sessions)
        
        xx = np.arange(sample_per_session)
        palette = sns.color_palette("husl", sample_per_session)
        
        # Now loop through the lists of positions and sessions
        for p, sec_id in zip(p_list, sec_ids):
            
            if with_knn: # Compute KNN neighbors for the last time step
                knn = self.get_knn_neighbors(
                    x=all_data['drone_speed'][:, -1, p].reshape(-1, 1, 1), 
                    y=all_data['pred_speed'][:, -1, p].reshape(-1, 1, 1)
                )
                knn_by_session = [x.squeeze().numpy() for x in np.split(knn, num_sessions)]
                
            for s, sim_id in zip(s_list, sim_ids):
                # Extract data for this specific position and session
                pdf_matrix = GMMPredictionHead.get_mixture_density(
                    rearrange(mixing_by_session[s][:, p], "T K -> () T () K"),
                    rearrange(means_by_session[s][:, p], "T K -> () T () K"),
                    rearrange(logvar_by_session[s][:, p], "T K -> () T () K"),
                    y_vals,
                ).squeeze().numpy()
                
                fig, ax = plt.subplots(figsize=(6, 4))
                for t in range(sample_per_session):
                    x_baseline = t  # the left side (time coordinate) for this ridge
                    ridge_x = t + self.density_scale * pdf_matrix[t, :]  # the right edge, shifted by density
                    ax.fill_betweenx(y_vals, x_baseline, ridge_x, color=palette[t], alpha=0.6)
                    
                    # create a scatter point plots for the KNN empirical distribution
                    if with_knn:
                        ax.scatter(
                            torch.ones(self.knn_nb) * x_baseline, 
                            knn_by_session[s][t], 
                            color='grey', s=5, alpha=0.5
                        )

                # Plot predictions and ground truth
                ax.plot(xx, pred_by_session[s][:, p], 'o-', label='30min Pred')
                ax.plot(xx, gt_by_session[s][:, p], 'x-', label='Ground Truth')
                ax.set_xticks(xx, xx * self.data_time_step + self.input_window)
                ax.set_xlabel("Time in Simulation")
                ax.set_ylabel("Speed (m/s)")
                ax.legend()
                
                fig.tight_layout()
                # Generate descriptive filename
                position_label = f"region{p}" if task == "reg" else f"section{p}_aimsun{sec_id}"
                
                save_path = f"{subfolder_path}/{position_label}_sim{sim_id}.pdf"
                fig.savefig(save_path)
                if verbose:
                    logger.info(f"Save GMM prediction visualization to {save_path}")
                plt.close(fig)  # Close figure to free memory
                
                
    def plot_pred_fix_time(
        self,
        all_preds,
        all_data,
        p_list,
        s_list,
        time_step_to_viz=15,
        pred_horizons=10,
        sample_per_session=20,
        task=None,
        sec_ids: List = None,
        sim_ids: List = None,
        subfolder_path=None,
        verbose=False,
    ):
        """ 
            Plot the GMM prediction at a fixed time stamp for different prediction horizons (3min to 30min ahead).
            E.g., for 8 am in simulation session 1, we plot the predicted speed when the input window is 3 min, 6 min, ..., 30 min before 8 am. 
            When the input window is 3 min before 8 am, our input data is from 7:27 - 7:57 am, and 8 am is just the next time step.

            Args:
                all_preds: dictionary of all predictions, values are tensors of shape (N, T, P, K) for GMM parameters and (N, T, P) for predicted speed
                all_data: dictionary of all data sequences (input and output alike), values are tensors of shape (N, T_i, P) where T_i may vary
                p_list: list of segment indices
                s_list: list of session indices 
                sample_per_session: sample per session, 20 for simbarca
                task: 'seg' or 'reg' to specify which task to plot
                sec_ids: list of section IDs in aimsun, will be used in file name
                sim_ids: list of simulation session IDs, will be used in file name
                time_step_to_viz: the time step index within each session to visualize (0 to sample_per_session-1)
        """
        # Determine which task to use
        assert task in self.eval_tasks, f"Task should be one of {self.eval_tasks}"
        assert 0 <= time_step_to_viz <= sample_per_session - pred_horizons , f"fixed_time_step must be between 0 and {sample_per_session} - {pred_horizons} = {sample_per_session - pred_horizons} for a model with {pred_horizons} prediction horizons"
        
        ymin = self.data_min[task]
        ymax = self.data_max[task]
        label_seq = self.seq_labels_by_task[task]
        
        y_vals = torch.linspace(ymin, ymax, self.vis_pts)
        num_sessions = len(all_preds[label_seq]) // sample_per_session
        
        # Split arrays by session for efficiency, keeping all time steps
        pred_by_session = torch.tensor_split(all_preds[label_seq], num_sessions)
        gt_by_session = torch.tensor_split(all_data[label_seq], num_sessions)
        mixing_by_session = torch.tensor_split(all_preds[f"{task}_mixing"], num_sessions)
        means_by_session = torch.tensor_split(all_preds[f"{task}_means"], num_sessions)
        logvar_by_session = torch.tensor_split(all_preds[f"{task}_log_var"], num_sessions)
        
        # Create prediction horizon labels (3min, 6min, ..., 30min)
        xx = np.arange(pred_horizons)
        pred_horizon_labels = [f"{self.data_time_step*(i+1)}" for i in range(pred_horizons)][::-1]  # reverse order for visualization
        palette = sns.color_palette("viridis", pred_horizons)  # Different colors for different horizons
        
        # Now loop through the lists of positions and sessions
        for p, sec_id in zip(p_list, sec_ids):
            for s, sim_id in zip(s_list, sim_ids):
                
                fig, ax = plt.subplots(figsize=(8, 6))
                batch_indices = torch.arange(time_step_to_viz, time_step_to_viz + pred_horizons)
                timestep_indices = - torch.arange(1, pred_horizons + 1)  # from -1 to -pred_horizons
                gmm_density = GMMPredictionHead.get_mixture_density(
                    rearrange(mixing_by_session[s][batch_indices, timestep_indices, p, :], "T K -> () T () K"),
                    rearrange(means_by_session[s][batch_indices, timestep_indices, p, :], "T K -> () T () K"),
                    rearrange(logvar_by_session[s][batch_indices, timestep_indices, p, :], "T K -> () T () K").exp(),
                    y_vals,
                ).squeeze().numpy()
                
                # create a ridge plot for the GMM density
                for t in range(pred_horizons):
                    x_baseline = t
                    ridge_x = t + self.density_scale * gmm_density[t, :]
                    
                    ax.fill_betweenx(y_vals, x_baseline, ridge_x, color=palette[t], alpha=0.6)
                # Plot predictions and ground truth
                ax.plot(
                    xx, 
                    pred_by_session[s][batch_indices, timestep_indices, p], 
                    'o-', label='Predictions'
                )
                ax.plot(
                    xx, 
                    gt_by_session[s][batch_indices, timestep_indices, p], 
                    '-', label='Ground Truth'
                )
                # Add labels and legend
                ax.set_xlabel("Time ahead of Prediction")
                ax.set_ylabel("Speed (m/s)")
                ax.legend()
                ax.set_title(f"Predictions for Section {sec_id} in Session {sim_id} at Time {time_step_to_viz*self.data_time_step + self.input_window} min")
                ax.set_xticks(xx, pred_horizon_labels)
                fig.tight_layout()
                
                save_path = f"{subfolder_path}/{task}_{sec_id}_sim{sim_id}_time{time_step_to_viz*self.data_time_step + self.input_window}.pdf"
                fig.savefig(save_path)
                if verbose:
                    logger.info(f"Save GMM prediction visualization to {save_path}")
                plt.close(fig)