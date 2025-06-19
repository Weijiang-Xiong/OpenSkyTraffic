import os
import logging
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

import pandas as pd
import geopandas
import contextily as ctx

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

from netsanut.evaluation import collect_predictions, uncertainty_metrics, EVAL_CONFS
from netsanut.utils.io import make_dir_if_not_exist

sns.set_style("darkgrid")

class Visualizer:
    
    def __init__(self, model:nn.Module, save_dir="./") -> None:
        self.model = model
        self.model.eval()
        self.save_dir = "{}/visualize/".format(save_dir)
        make_dir_if_not_exist(self.save_dir)
        self.offset_coeffs = {c:self.model.offset_coeff(confidence=c) for c in EVAL_CONFS}
        # self.result_dict: Dict[str, torch.Tensor]
    
    def inference_on_dataset(self, dataloader: DataLoader):
        result_dict = collect_predictions(self.model, dataloader)
        self.src = result_dict['source'][..., 0]
        self.src_tid = result_dict['source'][..., 1]
        self.scale = result_dict['sigma']
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

    def calculate_metrics(self, verbose=True, per_loc=False):
        
        res = uncertainty_metrics(self.pred, self.target, self.scale, 
                                  offset_coeffs=self.offset_coeffs,
                                  ignore_value=0.0,
                                  verbose=verbose)
        if per_loc:
            res = [   
                uncertainty_metrics(self.pred[:, :, s], self.target[:, :, s], self.scale[:, :, s], 
                                  offset_coeffs=self.offset_coeffs,
                                  ignore_value=0.0,
                                  verbose=verbose)
                for s in range(self.pred.shape[-1])]

        return res
    
    @staticmethod
    def visualize_calibration(res, save_dir="./", save_hint=None):
        
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
        """ The purpose of calibration is to adjust the confidence intervals, so that the data
        coverage matches the confidence level, e.g., a 50% confidence interval should cover 50% 
        of the data points on average, no more, no less, just exactly 50%.
        """
        confidences = np.round(np.arange(0.05, 1.0, 0.05), 2).tolist() + [0.99, 0.999, 0.9999, 0.99999]
        offset_coeffs = {c:self.model.offset_coeff(c) for c in confidences}
        
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
    
    @staticmethod
    def visualize_map(res: List[Dict], metadata: Dict, save_dir="./", save_hint:str=None):
        """ draw mAO, mCO, mCCE for each sensor on the map, scatter plot (mCP, mCCE)
        
        Args:
            res (List[Dict]): uncertainty metrics per sensor, use self.calculate_metric(per_loc=True)
            metadata (Dict): metadata with geographical location of the sensors
            save_dir (str, optional): Defaults to "./".
            save_hint (str, optional): notes to add to file name Defaults to None.
        """
        summary_dict = {metric: [r[metric] for r in res] for metric in ['mAO', "mCP", "mCCE"]}
        
        fig, ax = plt.subplots(figsize=(5,4))
        ax.scatter(summary_dict['mCCE'], summary_dict['mCP'], s=3)
        ax.set_xlabel("mCCE")
        ax.set_ylabel("mCP")
        ax.set_title("mCCE and mCP per sensor")
        fig.tight_layout()
        
        if os.path.exists(save_dir):
            file_name = "mCCE_mCP" if save_hint is None else "mCCE_mCP_{}".format(save_hint)
            fig.savefig("{}/{}.pdf".format(save_dir, file_name))

        geo_locations = metadata["geo_loc"]

        west = min(geo_locations[:, 0])
        east = max(geo_locations[:, 0])
        south = min(geo_locations[:, 1])
        north = max(geo_locations[:, 1])
        
        dlong = abs(east - west)
        dlat = abs(north - south)
        
        # plot 2 transparent points in the map, to enlarge the plotted area a bit 
        # https://geopandas.org/en/stable/gallery/create_geopandas_from_pandas.html
        df_b = pd.DataFrame(
            {
                "Points": ["Boundary1", "Boundary2"],
                "Longitude": [west - 0.1*dlong, east+0.1*dlong],
                "Latitude": [south - 0.1*dlat, north+0.1*dlat],
            }
        )
        gdf_b = geopandas.GeoDataFrame(
            df_b, geometry=geopandas.points_from_xy(df_b.Longitude, df_b.Latitude), crs="EPSG:4326"
        ).to_crs(epsg=3857) # transform all points from longitude-latitude coordinates to Web Mercator

        # now create a new df with the sensors and plot their locations
        # remember to use .to_csr with geopandas
        df = pd.DataFrame(
            {
                "Points": ["sensor{}".format(n) for n in range(len(geo_locations))],
                "Longitude": [longitude for longitude, latitude in geo_locations],
                "Latitude": [latitude for longitude, latitude in geo_locations],
            }
        )
        gdf = geopandas.GeoDataFrame(
            df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326"
        ).to_crs(epsg=3857) # transform all points from longitude-latitude coordinates to Web Mercator
        # coordinates in Spherical Mercator projection
        points = np.array([(p.x, p.y) for p in gdf.geometry])
        xs, ys = points[:, 0], points[:, 1]

        for metric in ["mAO", "mCCE", "mCP"]:
            # https://geopandas.org/en/stable/gallery/plotting_basemap_background.html
            fig, ax = plt.subplots()
            gdf_b.plot("Points", ax=ax, markersize=0) # just to set the boundary

            # the metrics are indicated using color
            ctx.add_basemap(ax)
            scatter_handle = ax.scatter(xs, ys, c=summary_dict[metric])
            cbar = fig.colorbar(scatter_handle, shrink=0.7) 
            cbar.ax.set_title(metric)
            
            fig.tight_layout()
            ax.axis("off")
            if os.path.exists(save_dir):
                file_name = "plot_map_{}".format(metric)
                file_name = file_name if save_hint is None else "{}_{}".format(file_name, save_hint)
                fig.savefig("{}/{}.pdf".format(save_dir, file_name))