import logging
import numpy as np 
from typing import Dict

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader

from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style('darkgrid')

from .simbarca_evaluation import SimBarcaEvaluator

logger = logging.getLogger("default")

class SimBarcaGMMEvaluator(SimBarcaEvaluator):
    
    def __init__(self, ignore_value=-1.0, mape_threshold=1.0, save_dir: str=None, save_res: bool=True, save_note="", visualize=False, add_output_seq=[]) -> None:
        super().__init__(ignore_value, mape_threshold, save_dir, save_res, save_note, visualize)
        self.add_output_seq = add_output_seq
        
    def evaluate(self, model: nn.Module, data_loader: DataLoader, verbose=False) -> Dict[str, float]: 
        super().evaluate(model, data_loader, verbose=verbose)
        dataset = data_loader.dataset
        output_seqs = dataset.output_seqs
        sections_of_interest = dataset.sections_of_interest
        s2i = {v:k for k, v in dataset.index_to_section_id.items()} # section ID to array index in space dimension
        session_ids, demand_scales = dataset.get_session_properties(fit_dataset_len=False)
        
        all_preds, all_labels = self.collect_predictions(model, data_loader, output_seqs, output_seqs + self.add_output_seq)
        for s in range(26):
            self.plot_gmm_predictions(all_preds, all_labels, p=s2i[9971], s=s, save_note="sec{}_sim{}".format(9971, session_ids[s]))
            for p in range(4):
                self.plot_gmm_predictions(all_preds, all_labels, p, s, regional=True, save_note="sim{}".format(session_ids[s]))
    
    def plot_gmm_predictions(self, all_preds, all_labels, p, s, T=20, regional=False, save_note=""):
        """ p: segment index
            s: session index 
            T: sample per session, 20 for simbarca
        """
        density_scale = 5
        if not regional:
            ymin, ymax = 0, 14
            label_seq = "pred_speed"
            gmm_seqs_prefix = "seg"
        else:
            ymin, ymax = 0, 9
            label_seq = "pred_speed_regional"
            gmm_seqs_prefix = "reg"
        
        def obtain_gmm_pdf(mixing, means, variance, y_vals):
            
            # Compute the GMM PDF at each time step over y_vals
            pdf_matrix = np.zeros((T, len(y_vals)))
            for t in range(T):
                for k in range(K):
                    pdf_matrix[t, :] += mixing[t, k] * norm.pdf(y_vals, 
                                                                loc=means[t, k],
                                                                scale=np.sqrt(variance[t, k]))

            return pdf_matrix
        
        y_vals = np.linspace(ymin, ymax, 300)
        num_sessions = len(all_preds[label_seq]) // T
        
        # put everything to numpy
        all_preds = {k: v.numpy() for k, v in all_preds.items()}
        all_labels = {k: v.numpy() for k, v in all_labels.items()}
        
        pred_by_session = np.split(all_preds[label_seq][:, -1, p], num_sessions)
        gt_by_session = np.split(all_labels[label_seq][:, -1, p], num_sessions)
        mixing_by_session = np.split(all_preds["{}_mixing".format(gmm_seqs_prefix)][:, -1, p], num_sessions)
        means_by_session = np.split(all_preds["{}_means".format(gmm_seqs_prefix)][:, -1, p], num_sessions)
        variance_by_session = np.split(all_preds["{}_log_var".format(gmm_seqs_prefix)][:, -1, p], num_sessions)
        
        _, K = mixing_by_session[0].shape
        xx = np.arange(T)
        palette = sns.color_palette("husl", T)
        pdf_matrix = obtain_gmm_pdf(mixing_by_session[s], means_by_session[s], np.exp(variance_by_session[s]), y_vals)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        for t in range(T):
            x_baseline = t  # the left side (time coordinate) for this ridge
            ridge_x = t + density_scale * pdf_matrix[t, :]  # the right edge, shifted by the (scaled) density
            plt.fill_betweenx(y_vals, x_baseline, ridge_x, color=palette[t], alpha=0.6)
            
        plt.plot(xx, pred_by_session[s], 'o-', label='30min Pred')
        plt.plot(xx, gt_by_session[s], 'x-', label='Ground Truth')
        # set x axis and y axis name
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Speed (m/s)")
        ax.legend()
        
        plt.tight_layout()
        save_note = save_note if not regional else save_note + "_regional"
        save_path = "{}/vis_gmm_pred_p{}_s{}_{}.pdf".format(self.save_dir, p, s, save_note)
        plt.savefig(save_path)
        logger.info("Save GMM prediction visualization to {}".format(save_path))