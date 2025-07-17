import json
import logging
import numpy as np 

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns 

from typing import Dict, List, Tuple
from collections import defaultdict

from ..utils.io import flatten_results_dict, make_dir_if_not_exist
from .metrics import common_metrics
from ..data.datasets import SimBarcaMSMT

sns.set_style('darkgrid')
logger = logging.getLogger("default")

class SimBarcaEvaluator:

    def __init__(self, ignore_value=float("nan"), mape_threshold=1.0, save_dir: str=None, visualize=False) -> None:
        self.ignore_value = ignore_value
        self.mape_threshold = mape_threshold
        self.save_dir = save_dir
        self.visualize = visualize
        make_dir_if_not_exist(self.save_dir)
        # for saving various evaluation metrics in analysis
        self.metrics_scalar: Dict[str, float] = dict()
        self.metrics_vector: Dict[str, List] = dict()
        
    def __call__(self, model: nn.Module, data_loader: DataLoader, **kwargs) -> Dict[str, float]:
        return self.evaluate(model, data_loader, **kwargs)

    def collect_predictions(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        pred_seqs: List = None,
        data_seqs: List = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """run inference of the model on the data_loader concatenate all predictions and corresponding labels.
        
        The sequences in `data_seqs` will be collected from the data batch. The sequences in `pred_seqs` will be collected from the model output. In simple cases, they are the same, but when the evaluation relies on additional sequences, they can be different as well (for example in uncertainty evaluation).
        """
        model.eval()

        all_preds = defaultdict(list)
        all_labels = defaultdict(list)

        for data_dict in data_loader:
            with torch.no_grad():
                pred_dict = model(data_dict)
            
            # collect predicted sequences corresponding labels
            for name in pred_seqs:
                all_preds[name].append(pred_dict[name])
            for name in data_seqs: 
                all_labels[name].append(data_dict[name])
                
        # this will actually modify all_preds and all_labels
        for res_collection in [all_preds, all_labels]:
            for key, value in res_collection.items():
                res_collection[key] = torch.cat(value, dim=0).detach().cpu()
        
        return all_preds, all_labels

    def mae_by_demand_scale(self, all_preds, all_labels, demand_scales):
        """ compute the mean absolute error for each sample and group by demand scales, then create a boxplot for each demand scale

        Args:
            all_preds: Dict[str, torch.Tensor] with keys as sequence names and values as predicted sequences, each sequence should have shape (N, T, P) with T and P being the time step and the spatial locations.
            all_labels: Dict[str, torch.Tensor] with keys as sequence names and values as labels, same shape as `all_preds`.
            demand_scales: torch tensor with shape (N,) indicating the demand scale of each sample
        """
        
        for seq_name in all_labels.keys():
            seq_preds, seq_labels = all_preds[seq_name], all_labels[seq_name]
            abs_error = torch.nanmean(torch.abs(seq_preds - seq_labels), dim=(1, 2))
            error_by_scale = dict()
            for scale in demand_scales.unique():
                scale_idx = demand_scales == scale
                error_by_scale[round(scale.item(), 2)] = abs_error[scale_idx].numpy()
        
            # plot the boxplot
            fig, ax = plt.subplots()
            ax.boxplot(error_by_scale.values(), labels=error_by_scale.keys())
            ax.set_xlabel("Demand Scale")
            ax.set_ylabel("MAE")
            # ax.set_title("MAE by Demand Scale for {}".format(seq_name))
            fig_path = "{}/MAE_by_demand_scale_{}.pdf".format(self.save_dir, seq_name)
            fig.tight_layout()
            fig.savefig(fig_path)
            logger.info("Save MAE by demand scale plot to {}".format(fig_path))
            plt.close()
            
    def segment_mae_by_avg_speed(self, all_preds, all_labels, step_size=1):
        """ plot the MAE of each segment by the average speed of the segment
        """
        
        preds, labels = all_preds["pred_speed"], all_labels["pred_speed"]
        avg_spd_per_segment = torch.nanmean(labels, dim=(0, 1))
        # discard nan values (i.e., segments with no vehicle ever passed)
        avg_spd_per_segment = avg_spd_per_segment[~torch.isnan(avg_spd_per_segment)]
        abs_error_per_segment = torch.nanmean(torch.abs(preds - labels), dim=(0, 1))
        # discard nan values (i.e., segments with no vehicle ever passed)
        abs_error_per_segment = abs_error_per_segment[~torch.isnan(abs_error_per_segment)]
        
        # we define the bin edges based on the average speed of the segments
        # round to the nearest multiple of step_size
        bin_min = np.round(avg_spd_per_segment.min() / step_size) * step_size
        if bin_min == 0:
            bin_min = step_size
        bin_max = np.round(avg_spd_per_segment.max() / step_size) * step_size
        # bin_max is included, and only once.
        avg_spd_bins = np.arange(bin_min, bin_max, step_size)
        avg_spd_bins = np.append(avg_spd_bins, bin_max)
        
        # which bin each segment belongs to
        avg_spd_bin_idx = np.digitize(avg_spd_per_segment, avg_spd_bins) 
        # calculate the average MAE for each bin
        mae_by_avg_spd = dict()
        for i in range(1, len(avg_spd_bins)):
            idx = avg_spd_bin_idx == i
            mae_by_avg_spd[avg_spd_bins[i]] = abs_error_per_segment[idx].numpy()

        # plot the boxplot
        fig, ax = plt.subplots()
        ax.boxplot(mae_by_avg_spd.values(), labels=mae_by_avg_spd.keys())
        ax.set_xlabel("Upper bound of average segment speed")
        ax.set_ylabel("MAE")
        # ax.set_title("MAE by Average Speed")
        fig_path = "{}/MAE_by_avg_speed.pdf".format(self.save_dir)
        fig.tight_layout()
        fig.savefig(fig_path)
        plt.close()
        
    def eval_by_time_step(self, all_preds, all_labels, verbose=False):
        logger = logging.getLogger("default")
        
        eval_res_over_time = defaultdict(lambda: defaultdict(list))
        for seq_name in all_labels.keys():
            if verbose:
                logger.info("Evaluate model on {}".format(seq_name))

            seq_preds, seq_labels = all_preds[seq_name], all_labels[seq_name]

            # evaluate each predicted time step, i.e., forecasting from 5 min up to 1 hour
            for i in range(seq_preds.shape[1]):  # number of predicted time step
                pred = seq_preds[:, i, :]
                real = seq_labels[:, i, :]
                step_metrics = common_metrics(pred, real, self.ignore_value, self.mape_threshold)
                for k, v in step_metrics.items():
                    eval_res_over_time[seq_name][k].append(v)
                    
                if verbose:
                    logger.info("Evaluate model on test data at {:d} time step".format(i + 1))
                    logger.info(
                        "MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}".format(
                            step_metrics["mae"], step_metrics["mape"], step_metrics["rmse"]
                        )
                    )

            # average performance on all prediction steps, usually not reported in papers, but this can be useful in logging
            seq_res = common_metrics(seq_preds, seq_labels, self.ignore_value, self.mape_threshold)
            if verbose:
                logger.info("On average over different time steps")
                logger.info(
                    "MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}".format(seq_res["mae"], seq_res["mape"], seq_res["rmse"])
                )

            # # evaluate uncertainty prediction if possible
            # if getattr(model, 'is_probabilistic', False):
            #     all_scales = all_preds['sigma']
            #     offset_coeffs = {c:model.offset_coeff(confidence=c) for c in EVAL_CONFS}
            #     res_u = uncertainty_metrics(all_preds, all_labels, all_scales, offset_coeffs, verbose=verbose)
            #     res.update(res_u)
        if verbose:
            logger.info("Results to copy in manuscripts: \n {} \n".format(self.format_results_latex(eval_res_over_time)))
            logger.info("Results to copy in excel: \n {} \n".format(self.format_results_excel(eval_res_over_time)))
        
        # compute results per time step and over all time steps
        eval_res_over_time = flatten_results_dict(eval_res_over_time)
        avg_eval_res = {k:sum(v)/len(v) for k, v in flatten_results_dict(eval_res_over_time).items()}
        
        # save the results to self.metrics_vector and self.metrics_scalar
        self.metrics_vector.update(eval_res_over_time)
        self.metrics_scalar.update(avg_eval_res)


    def evaluate(self, model: nn.Module, data_loader: DataLoader, verbose=False) -> Dict[str, float]:
        dataset: SimBarcaMSMT = data_loader.dataset
        seqs = dataset.output_seqs
        sections_of_interest = dataset.sections_of_interest
        session_ids = torch.tensor(dataset.session_ids)  
        demand_scales = torch.tensor(dataset.demand_scales)
        
        all_preds, all_labels = self.collect_predictions(model, data_loader, seqs, seqs)

        self.eval_by_time_step(all_preds, all_labels, verbose=verbose)
        
        # this is for evaluation after the training process is done
        if self.visualize:
            self.save_scores_to_json()
            # visualize average MAE per location as a map
            self.plot_MAE_by_location(
                dataset.node_coordinates,
                all_preds,
                all_labels,
            )
            
            # do evaluation for different demand scales
            self.mae_by_demand_scale(
                all_preds, 
                all_labels, 
                torch.repeat_interleave(
                    demand_scales,
                    repeats=dataset.sample_per_session
                )
            )
            # evaluation by segment average speed
            self.segment_mae_by_avg_speed(all_preds, all_labels)
            
            self.pred_save_dir = "{}/predictions".format(self.save_dir)
            make_dir_if_not_exist(self.pred_save_dir)
            
            logger.info("Plotting predictions for sections of interest and regional predictions, check the plots in {}".format(self.pred_save_dir))
            section_id_to_index = {v:k for k, v in dataset.index_to_section_id.items()}
            for section_id in sections_of_interest:
                section_index = section_id_to_index[section_id]
                aimsun_sec_id = dataset.index_to_section_id[section_index]

                self.plot_pred_for_section(
                    all_preds,
                    all_labels,
                    session_ids,
                    demand_scales,
                    section_num=section_index,
                    save_note="_aimsunid_{}".format(aimsun_sec_id),
                )
            # plot for regional prediction
            for r in range(4):
                self.plot_pred_for_section(
                    all_preds,
                    all_labels,
                    session_ids,
                    demand_scales,
                    section_num=r,
                    regional=True,
                    save_note="region{}".format(r),
                )
        # return this for EvalHook to do logging in training
        return self.metrics_scalar


    def plot_pred_for_section(self, all_preds, all_labels, session_ids, demand_scales, section_num=100, regional=False, save_note="example", verbose=False):

        p = section_num
        sample_per_session = 20
        nrows, ncols = 4, 6
        sequence = 'pred_speed' if not regional else 'pred_speed_regional'
        total_num_session = int(all_preds[sequence].shape[0] / sample_per_session)
        
        pred_by_session = np.split(all_preds[sequence][:, -1, p], total_num_session)
        gt_by_session = np.split(all_labels[sequence][:, -1, p], total_num_session)
        xx = np.arange(sample_per_session)
        
        sessions_to_include = list(range(int(total_num_session)))[-nrows * ncols:] # id in split
        session_ids = session_ids[sessions_to_include]
        demand_scales = demand_scales[sessions_to_include]
        
        fig, ax = plt.subplots(nrows, ncols, figsize=(13, 5))
        for i in range(nrows):
            for j in range(ncols):
                idx = sessions_to_include[i * ncols + j]
                idx_dataset = session_ids[i * ncols + j]
                ds = demand_scales[i * ncols + j]
                ax[i, j].plot(xx, pred_by_session[idx], label='30min_pred')
                ax[i, j].plot(xx, gt_by_session[idx], label='GT')
                ax[i, j].set_title("Sim. {} D.S. {}".format(idx_dataset, round(ds.item(), 2)), fontsize=0.8*plt.rcParams['font.size'])
        
        # add a common legend for all the subplots
        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncols=2)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the legend at the top
        fig.savefig("{}/30min_ahead_{}_{}_{}.pdf".format(self.pred_save_dir, sequence, p, save_note))
        if verbose:
            logger.info("Saved the plot to {}/30min_ahead_{}_{}_{}.pdf".format(self.pred_save_dir, sequence, p, save_note))
        
        
    def plot_MAE_by_location(self, node_coordinates, all_preds, all_labels, save_note="example"):
        MAE = torch.abs(all_preds['pred_speed'] - all_labels['pred_speed'])
        mae_by_section = torch.nanmean(MAE, dim=(0,1))
        fig, ax = plt.subplots(figsize=(6.5, 4))
        im = ax.scatter(node_coordinates[:, 0], node_coordinates[:, 1], c=mae_by_section.numpy(), s=2)
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("X Coordinates")
        ax.set_ylabel("Y Coordinates")
        fig.tight_layout()
        fig.savefig("{}/average_mae_{}.pdf".format(self.save_dir, save_note))
        logger.info("Saved the plot to {}/average_mae_{}.pdf".format(self.save_dir, save_note))
    
    
    def format_results_excel(self, eval_res):
        excel_string = "METHOD"
        for task in eval_res.keys():
            for time_step in [4, 9]:
                excel_string += " {:.2f} {:.1f}% {:.2f}".format(
                    eval_res[task]["mae"][time_step],
                    eval_res[task]["mape"][time_step]*100,
                    eval_res[task]["rmse"][time_step]
                )
        return excel_string
        
    def format_results_latex(self, eval_res):
        latex_string = "METHOD"
        for task in eval_res.keys():
            for time_step in [4, 9]:
                latex_string += " & {:.2f} & {:.1f}\\% & {:.2f}".format(
                    eval_res[task]["mae"][time_step],
                    eval_res[task]["mape"][time_step]*100,
                    eval_res[task]["rmse"][time_step]
                )
        latex_string += " \\\\"
        return latex_string
    
    def save_scores_to_json(self, file_name: str = "final_evaluation_scores.json"):
        """
        Save the scores to a JSON file.
        The scores are saved in a dictionary with keys being the score types and values being the scores.
        """

        scalar_res = {k:float(v) for k, v in self.metrics_scalar.items()}
        vector_res = {k:v for k, v in self.metrics_vector.items() if isinstance(v, list)}
        res_to_save = {
            "average": scalar_res,
            "horizon": vector_res
        }

        save_path = f"{self.save_dir}/{file_name}"
        with open(save_path, 'w') as f:
            json.dump(res_to_save, f, indent=4)
        logger.info(f"Saved scores to {save_path}")