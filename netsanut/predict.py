import os
import logging
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style("darkgrid")

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

from netsanut.evaluation import inference_on_dataset, uncertainty_metrics, GGD_interval

class Visualizer:
    
    def __init__(self, model:nn.Module, dist_exponent=2, save_dir="./") -> None:
        self.model = model
        self.model.eval()
        self.save_dir = save_dir
        # this part should be added to model code later
        self.dist_exponent = dist_exponent
        self.offset_coeffs = {c:GGD_interval(beta=self.dist_exponent, confidence=c) 
                         for c in np.round(np.arange(0.5, 1.0, 0.05), 2).tolist()}
        # self.result_dict: Dict[str, torch.Tensor]
    
    def inference_on_dataset(self, dataloader: DataLoader):
        result_dict = inference_on_dataset(self.model, dataloader)
        self.src = result_dict['source'][..., 0]
        self.src_tid = result_dict['source'][..., 1]
        # self.scale = torch.sqrt(result_dict['logvar'].exp())
        self.scale = torch.pow(result_dict['logvar'].exp(), 1.0/self.dist_exponent)
        self.pred = result_dict['pred']
        self.target = result_dict['target']
        
    def visualize_day(self, conf=0.95, start_idx=200, pred_step=11, sensor=100, num_days=1, save_name=None):
        
        length =int(24*60/5)
        k = self.offset_coeffs[conf]
        
        xs = np.arange(length)/12
        gt = self.target[start_idx:start_idx+length][:, pred_step, sensor]
        ys = self.pred[start_idx:start_idx+length][:, pred_step, sensor]
        ub = ys + k * self.scale[start_idx:start_idx+length][:, pred_step, sensor]
        lb = ys - k * self.scale[start_idx:start_idx+length][:, pred_step, sensor]
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(xs, ys, label="Pred.")
        ax.plot(xs, gt, label="GT")
        ax.fill_between(xs, ub, lb, label="95 Int.", alpha=0.5)
        ax.set_xlim(0, 24)
        ax.set_xticks(list(range(0, 24, 4)) + [24])
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("Value")
        ax.legend()
        fig.tight_layout()
        
        if os.path.exists(self.save_dir):
            fig.savefig("{}/{}.pdf".format(self.save_dir, save_name))

    def calculate_metrics(self, verbose=True):
        
        res = uncertainty_metrics(self.pred, self.target, self.scale, 
                                  offset_coeffs=self.offset_coeffs,
                                  ignore_value=0.0,
                                  verbose=verbose)
        return res
    
    @staticmethod
    def visualize_calibration(res, save_dir, save_hint=None):
        
        xs = np.round(np.arange(0.5, 1.0, 0.05), 2)
        ys = res['coverage_percentage']
        
        fig, ax = plt.subplots(figsize=(4,4))
        ax.plot(xs, ys, label="Model")
        ax.plot(xs, xs, "--", label="Ideal")
        ax.set_xlim(0.5, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence Interval")
        ax.set_ylabel("Data Coverage")
        ax.set_title("Uncertainty Calibration \n mAO {:.3f}, mCCE {:.3f}".format(res['mAO'], res['mCCE']))
        ax.legend()
        fig.tight_layout()
        
        if os.path.exists(save_dir):
            file_name = "calibration_curve" if save_hint is None else "calibration_curve_{}".format(save_hint)
            fig.savefig("{}/{}.pdf".format(save_dir, file_name))
    
    def calibrate_scale_offset(self, verbose=True):
        
        confidences = np.round(np.arange(0.05, 1.0, 0.05), 2).tolist() + [0.99, 0.999, 0.9999, 0.99999]
        offset_coeffs = {c:GGD_interval(beta=self.dist_exponent, confidence=c) 
                         for c in confidences}
        
        init_res = uncertainty_metrics(self.pred, self.target, self.scale, 
                                  offset_coeffs=offset_coeffs,
                                  ignore_value=0.0,
                                  verbose=verbose)
        xp, fp = init_res['coverage_percentage'], list(offset_coeffs.values())
        calibrated_coeffs = {x:np.interp(x, xp, fp) for x in np.round(np.arange(0.5, 1.0, 0.05), 2)}
        
        res = uncertainty_metrics(self.pred, self.target, self.scale, 
                                  offset_coeffs=calibrated_coeffs,
                                  ignore_value=0.0,
                                  verbose=verbose)
        
        return res
        
    def visualize_attention(self):
        
        pass
    
    def visualize_map(self):
        pass 
    

        
    