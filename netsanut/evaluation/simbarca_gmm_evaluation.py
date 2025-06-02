import json
import logging
import numpy as np 
from typing import Dict, List

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns 

from einops import rearrange

from ..utils.io import make_dir_if_not_exist
from ..models.gmmpred import GMMPredictionHead
from ..data.datasets import SimBarca
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
        self.sp_size = 10  # the size of the chunks to split the tensors space dimension, default 20
        self.knn_nb = 20  # the number of nearest neighbors to find, default 20
        self.gpu = True  # whether to use GPU acceleration, default True

    
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
        """ Given an error score tensor of shape (N, T, P), we summarize the score over the time steps.
            The summary is saved to self.saved_metrics, with the key being the note.
            
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
    
    def evaluate(self, model: nn.Module, data_loader: DataLoader, verbose=False) -> Dict[str, float]:
        
        _ = super().evaluate(model, data_loader, verbose=verbose)
        
        dataset: SimBarca = data_loader.dataset
        soi: List[int] = dataset.sections_of_interest
        s2i = dataset.section_id_to_index  # section ID to array index in space dimension
        session_ids = dataset.session_ids  # list of simulation session IDs

        all_preds, all_data = self.collect_predictions(
            model,
            data_loader,
            pred_seqs=dataset.output_seqs + self.add_output_seq,
            data_seqs=dataset.output_seqs + ['drone_speed'],
        )
        cdf_xs = torch.linspace(self.data_min['seg'], self.data_max['seg'] , self.ci_pts)

        
        crps_gmm_emp = self.get_crps_gmm_vs_emp_dist(
            mixing=all_preds["seg_mixing"],
            means=all_preds["seg_means"],
            log_var=all_preds["seg_log_var"],
            xs=cdf_xs,
            inputs=all_data["drone_speed"],
            gt=all_data["pred_speed"],
            sp_size=self.sp_size,
            knn_nb=self.knn_nb,
            gpu=self.gpu,
        )
        self.analyze_scores(crps_gmm_emp, note="CRPS_GMM_EMP")
        if verbose:
            logger.info("Evaluate CRPS score for GMM predictions using KNN empirical distribution...")
            logger.info("The average CRPS {}".format(crps_gmm_emp.nanmean()))

        
        crps_pred_emp = self.get_crps_pred_vs_emp_dist(
            pred=all_preds["pred_speed"],
            xs=cdf_xs,
            inputs=all_data["drone_speed"],
            gt=all_data["pred_speed"],
            sp_size=self.sp_size,
            knn_nb=self.knn_nb,
            gpu=self.gpu,
        )
        self.analyze_scores(crps_pred_emp, note="CRPS_PRED_EMP")
        if verbose:
            logger.info("Evaluate CRPS score for point predictions using the KNN empirical distribution...")
            logger.info("The average CRPS {}".format(crps_pred_emp.nanmean()))
        
        crps_gmm_gt = self.get_crps_gmm_vs_gt(
            mixing=all_preds["seg_mixing"],
            means=all_preds["seg_means"],
            log_var=all_preds["seg_log_var"],
            xs=cdf_xs,
            gt=all_data["pred_speed"],
            sp_size=self.sp_size,
            gpu=self.gpu,
        )
        self.analyze_scores(crps_gmm_gt, note="CRPS_GMM_GT")
        if verbose:
            logger.info("Evaluate CRPS score for GMM predictions using the ground truth point distribution...")
            logger.info("The average CRPS {}".format(crps_gmm_gt.nanmean()))

        # for conf in self.eval_confs:
        #     self.eval_gmm_confidence_interval(
        #         mixing=all_preds["seg_mixing"],
        #         means=all_preds["seg_means"],
        #         log_var=all_preds["seg_log_var"],
        #         gt=all_labels["pred_speed"],
        #         xmin=self.data_min["seg"],
        #         xmax=self.data_max["seg"],
        #         n_points=self.ci_pts,
        #         conf=conf
        #     )
        
        if self.visualize:
            
            self.plot_scores()
            
            # Pass lists of positions and sessions instead of looping here
            p_list_seg = [s2i[sec] for sec in soi]
            s_list = list(range(len(session_ids)))
            p_list_reg = list(range(len(dataset.cluster_id.unique()))) # For regional predictions
            
            self.plot_pred_fix_time(all_preds, all_data,
                    p_list=p_list_seg, s_list=s_list,
                    time_step_to_viz=10, pred_horizons=10,
                    sample_per_session=dataset.sample_per_session, task="seg",
                    sec_ids=soi, sim_ids=session_ids)
            
            self.plot_30min_gmm_preds(all_preds, all_data, p_list=p_list_seg, s_list=s_list, 
                    sample_per_session=dataset.sample_per_session, task="seg", sim_ids=session_ids, sec_ids=soi, with_knn=True)
            self.plot_30min_gmm_preds(all_preds, all_data, p_list=p_list_reg, s_list=s_list,
                    sample_per_session=dataset.sample_per_session, task="reg", sim_ids=session_ids, sec_ids=p_list_reg)
    
        return self.saved_scores['scalar']


    def eval_gmm_confidence_interval(self, mixing, means, log_var, gt, xmin, xmax, n_points, conf):
        """
            Evaluate the GMM confidence interval for the predictions, with two aspects into account:
                1. The percentage of predictions within the confidence interval
                2. The width of the confidence interval
            
            mixing: the GMM mixing coefficients
            means: the GMM means
            variances: the GMM variances
            gt: the ground truth values
        """

        # split tensors along spatial dimension to save memory
        tensor_names = ["mixing", "means", "log_var"]
        tensors = [mixing, means, log_var]
        split_tensors = {}
        for name, tensor in zip(tensor_names, tensors):
            split_tensors[name] = torch.split(tensor, self.sp_size, dim=2)
        

        score_by_chunk = []
        # Get the number of chunks from any tensor (they all have the same number)
        num_chunks = len(split_tensors[tensor_names[0]])
        for i in range(num_chunks):
            # Extract current chunk for each tensor
            chunks = {name: split_tensors[name][i] for name in tensor_names}

            gmm_confidence_intervals = GMMPredictionHead.get_confidence_interval(
                chunks["mixing"].cuda(), chunks["means"].cuda(), chunks["log_var"].cuda(),
                xmin=xmin, xmax=xmax, n_points=n_points, conf=conf
            )
            
            # Compute the percentage of predictions within the confidence interval
            within_ci = (gmm_confidence_intervals[0] <= gt) & (gt <= gmm_confidence_intervals[1])
            score_by_chunk.append(within_ci.float())
    
        # Concatenate the chunks
        score = torch.cat(score_by_chunk, dim=-1).cpu()

        return score


    def _compute_crps(self, tensors, tensor_names, cdf_func1, cdf_func2, xs, sp_size=20, gpu=True):
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
            
            # Compute CDFs using the provided functions
            cdf1 = cdf_func1(chunks, xs, gpu)
            cdf2 = cdf_func2(chunks, xs, gpu)
            
            # Compute CRPS
            CRPS_chunk = torch.sum((cdf1 - cdf2)**2 * abs(xs[1] - xs[0]), dim=-1)
            CRPS_by_chunk.append(CRPS_chunk)
        
        # Concatenate the chunks
        CRPS = torch.cat(CRPS_by_chunk, dim=-1).cpu()
        
        return CRPS


    def seg_invalid_to_ignore_value(self, scores, gt):
        """
        Ignore invalid values in the scores tensor based on the ground truth tensor.
        
        Args:
            scores: Tensor of CRPS scores
            gt: Ground truth tensor
        """
        scores[gt.isnan()] = self.ignore_value

        return scores
    
    
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
        
        def gmm_cdf_func(chunks, xs, gpu):
            return self.get_gmm_cdf(chunks["mixing"], chunks["means"], chunks["log_var"], xs, gpu=gpu)
        def knn_ecdf_func(chunks, xs, gpu):
            return self.get_knn_ecdf(chunks["inputs"], chunks["gt"], knn_nb, xs, gpu=gpu)
        
        # split the tensors along the spatial location dimension to get chunks, save memory
        tensors = [mixing, means, log_var, inputs, gt]
        tensor_names = ["mixing", "means", "log_var", "inputs", "gt"]
        scores = self._compute_crps(tensors, tensor_names, gmm_cdf_func, knn_ecdf_func, xs, sp_size, gpu)
        
        return self.seg_invalid_to_ignore_value(scores, gt)
    
    
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
        def gmm_cdf_func(chunks, xs, gpu):
            return self.get_gmm_cdf(chunks['mixing'], chunks['means'], chunks['log_var'], xs, gpu)
        
        def gt_cdf_func(chunks, xs, gpu):
            return self.get_point_cdf(chunks['gt'], xs, gpu)
        
        tensors = [mixing, means, log_var, gt]
        tensor_names = ["mixing", "means", "log_var", "gt"]
        scores = self._compute_crps(tensors, tensor_names, gmm_cdf_func, gt_cdf_func, xs, sp_size, gpu)
        
        return self.seg_invalid_to_ignore_value(scores, gt)


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
        def pred_cdf_func(chunks, xs, gpu):
            return self.get_point_cdf(chunks['pred'], xs, gpu)
        
        def knn_cdf_func(chunks, xs, gpu):
            return self.get_knn_ecdf(chunks['inputs'], chunks['gt'], knn_nb, xs, gpu)
        
        tensors = [pred, inputs, gt]
        tensor_names = ["pred", "inputs", "gt"]
        scores = self._compute_crps(tensors, tensor_names, pred_cdf_func, knn_cdf_func, xs, sp_size, gpu)
        
        return self.seg_invalid_to_ignore_value(scores, gt)


    def get_point_cdf(self, tensor, xs, gpu=True):
        """ tensor has shape (N, T, sp_size), can be ground-truth or predicted most-likely values.
            xs has shape (X,), which is the points to evaluate the CDF.
        
            The CDF is a step function jumping at the value of tensor, with shape (N, T, sp_size, X).
        """
        if gpu and torch.cuda.is_available():
            tensor = tensor.cuda()
            xs = xs.cuda()
        
        point_cdf = (tensor.unsqueeze(-1) < xs).float()
        
        return point_cdf

        
    def get_gmm_cdf(self, mixing, means, log_var, xs, gpu=True):
        if gpu and torch.cuda.is_available():
            mixing = mixing.cuda()
            means = means.cuda()
            log_var = log_var.cuda()
            xs = xs.cuda()
        
        gmm_density = GMMPredictionHead.get_mixture_density(mixing, means, log_var.exp(), xs)
        dx = abs(xs[1] - xs[0]) # the step size of the x values
        gmm_cdf = torch.cumsum(gmm_density * dx, axis=-1)
        gmm_cdf = gmm_cdf / gmm_cdf[..., -1].unsqueeze(-1) # ensure the CDF ends at 1
        
        return gmm_cdf
    
    
    def get_knn_ecdf(self, inputs, gt, knn_nb, xs, gpu=True):
        if gpu and torch.cuda.is_available():
            inputs = inputs.cuda()
            gt = gt.cuda()
            xs = xs.cuda()
        
        gt_knn = self.compute_knn_neighbors(inputs, gt, k=knn_nb)
        
        sorted_gt, _ = torch.sort(gt_knn, dim=-1)
        # reshape sorted_gt to (N, T, sp_size, k, 1), so the last dimension will be broadcasted with xs
        knn_ecdf  = (sorted_gt.unsqueeze(-1) <= xs).sum(dim=-2).float() / knn_nb # → (N, T, sp_size, X)
        
        return knn_ecdf
    
    
    def compute_knn_neighbors(self, x, y, k=20):
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
        dist = torch.sqrt(torch.sum((x.unsqueeze(1) - x.unsqueeze(0))**2, dim=2))
        
        # Find k nearest neighbors for each sample at each P location
        # Result shapes: (N, k, P)
        _, knn_indices = torch.topk(dist, k=k, dim=1, largest=False)

        # Gather the corresponding y values using advanced indexing
        # https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing
        # Result shape: (N, k, P, T)
        knn_y = y[knn_indices, # from the k nearest neighbors
                  :, # keep all time steps 
                  torch.arange(P).view(1, 1, P).expand(N, k, P)]
        
        return rearrange(knn_y, 'N k P T -> N T P k')
    
    
    def plot_scores(self):
        """ 
            Plot the scores saved in self.saved_scores, which is a dictionary with keys being the score types.
            The values are numpy arrays with shape (T,) or (P,) for each time step and spatial location.
        """
        scores_by_time = self.saved_scores["vector"]
        # plot the scores by time step
        plot_legends = [
            ("CRPS_GMM_EMP", "GMM vs Empirical", "-"),
            ("CRPS_PRED_EMP", "Pred vs Empirical", "-"),
            ("CRPS_GMM_GT", "GMM vs GT", "--"),
            ("pred_speed_mae", "Pred vs GT (MAE)", "--"),
        ]

        fig, ax = plt.subplots(figsize=(6, 5))
        for seq_name, seq_legend, line_style in plot_legends:
            if seq_name in scores_by_time.keys():
                v = scores_by_time[seq_name]
            # plot the scores by time step using the specified line style
            ax.plot(v, line_style, label=seq_legend)
            
        ax.set_xticks(np.arange(len(v)))
        ax.set_xticklabels(3 * (np.arange(len(v))+1) )
        ax.set_xlabel("Prediction Time Horizion (min)")
        ax.set_ylabel("Score")
        ax.legend(loc="upper left")
        ax.set_title("CRPS by Time Horizion")
        plt.tight_layout()
        save_path = f"{self.save_dir}/scores_by_time_{self.save_note}.pdf"
        plt.savefig(save_path)
        logger.info(f"Save scores by time step to {save_path}")
        plt.close(fig)  # Close figure to free memory

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
                knn = self.compute_knn_neighbors(
                    x=all_data['drone_speed'][:, -1, p].reshape(-1, 1, 1), 
                    y=all_data['pred_speed'][:, -1, p].reshape(-1, 1, 1)
                )
                knn_by_session = [x.squeeze().numpy() for x in np.split(knn, num_sessions)]
                
            for s, sim_id in zip(s_list, sim_ids):
                # Extract data for this specific position and session
                pdf_matrix = GMMPredictionHead.get_mixture_density(
                    rearrange(mixing_by_session[s][:, p], "T K -> () T () K"),
                    rearrange(means_by_session[s][:, p], "T K -> () T () K"),
                    rearrange(logvar_by_session[s][:, p], "T K -> () T () K").exp(),
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
                
                # Create subfolder for 30min ahead predictions
                subfolder_path = f"{self.save_dir}/gmm_predictions"
                make_dir_if_not_exist(subfolder_path)
                
                save_path = f"{subfolder_path}/{position_label}_sim{sim_id}.pdf"
                fig.savefig(save_path)
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
    ):
        """ 
            Plot the GMM prediction at a fixed time stamp for different prediction horizons (3min to 30min ahead).
            E.g., for 8 am in simulation session 1, we plot the predicted speed when the input window is 3 min, 6 min, ..., 30 min before 8 am. When the input window is 3 min before 8 am, our input data is from 7:27 - 7:57 am, and 8 am is just the next time step.
            
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
                
                # Create subfolder for fixed time predictions
                subfolder_path = f"{self.save_dir}/fix_time_pred"
                make_dir_if_not_exist(subfolder_path)
                
                save_path = f"{subfolder_path}/{task}_{sec_id}_sim{sim_id}_time{time_step_to_viz*self.data_time_step + self.input_window}.pdf"
                fig.savefig(save_path)
                logger.info(f"Save GMM prediction visualization to {save_path}")
                plt.close(fig)