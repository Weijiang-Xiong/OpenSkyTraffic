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
from ..models.layers import GMMPredictionHead
from ..data.datasets import SimBarcaMSMT
from .simbarca_evaluation import SimBarcaEvaluator
from .metrics import (
    gmm_interval_coverage_and_width,
    ignore_score_when_gt_is,
    get_knn_neighbors,
    get_crps_pred_vs_emp_dist,
    get_crps_gmm_vs_emp_dist,
    get_crps_gmm_vs_gt
)

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
        visualize=False,
        add_output_seq: list = None,
    ) -> None:
        super().__init__(ignore_value, mape_threshold, save_dir, visualize)
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
        self.metrics_vector[note] = torch.nanmean(scores, dim=(0, 2)).cpu().numpy().tolist()
        # Save the mean over all the time steps, ignoring NaN values
        self.metrics_scalar[f'{note}'] = torch.nanmean(scores).item()
        
        if verbose:
            logger.info(f"Saved average {note} scores by time step to saved_metrics")
            logger.info(f"Overall average {note}: {self.metrics_scalar[f'{note}']:.4f}")
    

    #########################################################################################
    ################ Main evaluation routines           #####################################
    #########################################################################################
    def evaluate(self, model: nn.Module, data_loader: DataLoader, verbose=False) -> Dict[str, float]:
        """ This evaluation function is structured as follows:
                0. run the super class's evaluation function (deterministic metrics)
                1. collect predictions and required data sequences from the model and dataset
                2. run various evaluation functions (CRPS, CI, etc.), average scores saved to self.metrics_scalar and self.metrics_vector
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
            p_list_reg = list(range(len(np.unique(dataset.cluster_id)))) # For regional predictions
            
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
    
        return self.metrics_scalar


    #########################################################################################
    ################ Evaluation subroutines        ##########################################
    #########################################################################################
    def evaluate_confidence_interval(self, all_preds, all_data, verbose=False):
        """
        This function evaluates the confidence interval of the GMM prediction.
        """
        xs = torch.linspace(self.data_min["seg"], self.data_max["seg"], self.ci_pts)
        tensors = [all_preds["seg_mixing"], all_preds["seg_means"], all_preds["seg_log_var"], all_data["pred_speed"]]
        tensor_names = ["mixing", "means", "log_var", "gt"]

        for conf in self.eval_confs:
            within_ci, interval_width = gmm_interval_coverage_and_width(
                tensors=tensors,
                tensor_names=tensor_names,
                xs=xs,
                conf=conf,
                sp_size=self.sp_size,
                gpu=self.gpu,
            )

            within_ci = ignore_score_when_gt_is(within_ci, all_data["pred_speed"], self.ignore_value)
            interval_width = ignore_score_when_gt_is(interval_width, all_data["pred_speed"], self.ignore_value)
            self.analyze_scores(scores=within_ci.float(),note=f"CI_COVER_{conf}",verbose=False)
            self.analyze_scores(scores=interval_width,note=f"CI_WIDTH_{conf}",verbose=False)
        
        # from the saved CI_COVER and CI_WIDTH, compute the calibration error and average width
        # confidence calibration error is the absolute difference between confidence level and the average coverage
        CCE_conf_horizon, AW_conf_horizon = [], []
        # we evaluate CI by averaging the metrics over all prediction horizons. 
        # uncomment these following lines to save the CI evaluation results for each prediction horizon
        for conf in self.eval_confs: 
            CCE_conf_horizon.append([abs(x - conf) for x in self.metrics_vector[f'CI_COVER_{conf}']])
            AW_conf_horizon.append(self.metrics_vector[f'CI_WIDTH_{conf}'])

        # concatenate the scores along a new dimension, which is the confidence level
        mCCE_horizon = np.stack(CCE_conf_horizon, axis=-1).mean(axis=-1)
        mAW_horizon = np.stack(AW_conf_horizon, axis=-1).mean(axis=-1)

        # save the scores 
        self.metrics_scalar['mCCE'] = mCCE_horizon.mean().item()
        self.metrics_scalar['mAW'] = mAW_horizon.mean().item()
        self.metrics_vector['mCCE'] = mCCE_horizon.tolist()
        self.metrics_vector['mAW'] = mAW_horizon.tolist()
        if verbose:
            logger.info(f"mCCE: {self.metrics_scalar['mCCE']:.4f}, mAW: {self.metrics_scalar['mAW']:.4f}")


    def evaluate_crps(self, all_preds, all_data, verbose=False):

        cdf_xs = torch.linspace(self.data_min['seg'], self.data_max['seg'] , self.ci_pts)

        self.analyze_scores(
            scores=ignore_score_when_gt_is(
                get_crps_gmm_vs_emp_dist(
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
            scores=ignore_score_when_gt_is(
                get_crps_pred_vs_emp_dist(
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
            scores=ignore_score_when_gt_is(
                get_crps_gmm_vs_gt(
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
    ############### Functions for plotting statistics        ################################
    #########################################################################################

    def plot_crps_scores(self):
        """ 
            Plot the scores saved in self.saved_scores, which is a dictionary with keys being the score types.
            The values are numpy arrays with shape (T,) or (P,) for each time step and spatial location.
        """
        scores_by_time = self.metrics_vector
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
            save_path = f"{self.save_dir}/crps_by_time_{base_name}.pdf"
            plt.savefig(save_path)
            logger.info(f"Save scores by time step to {save_path}")
            plt.close(fig)  # Close figure to free memory


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
                knn = get_knn_neighbors(
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