import json
import logging
import numpy as np 

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style('darkgrid')

from typing import Dict
from collections import defaultdict

from netsanut.utils.io import flatten_results_dict, make_dir_if_not_exist
from netsanut.evaluation.metrics import prediction_metrics
from netsanut.data import SimBarca

class SimBarcaEvaluator:

    def __init__(self, ignore_value=-1.0, mape_threshold=1.0, save_dir: str=None, save_res: bool=True, save_note="example", visualize=False) -> None:
        self.ignore_value = ignore_value
        self.mape_threshold = mape_threshold
        self.save_dir = save_dir
        self.save_res = save_res
        self.save_note = save_note
        self.visualize = visualize
        make_dir_if_not_exist(self.save_dir)
    
    def __call__(self, model: nn.Module, data_loader: DataLoader, **kwargs) -> Dict[str, float]:
        return self.evaluate(model, data_loader, **kwargs)

    def collect_predictions(self, 
        model: nn.Module, data_loader: DataLoader, seq_names=[]
    ) -> Dict[str, torch.Tensor]:
        """run inference of the model on the data_loader
        concatenate all predictions and corresponding labels.

        Returns: predictions, labels
        """
        model.eval()

        all_preds = defaultdict(list)
        all_labels = defaultdict(list)

        for data_dict in data_loader:
            with torch.no_grad():
                pred_dict = model(data_dict)

            for name in seq_names: # collect predicted sequences and corresponding labels
                all_preds[name].append(pred_dict[name])
                all_labels[name].append(data_dict[name])

        # this will actually modify all_preds and all_labels
        for res_collection in [all_preds, all_labels]:
            for key, value in res_collection.items():
                res_collection[key] = torch.cat(value, dim=0).detach().cpu()
        
        return all_preds, all_labels


    def evaluate(self, model: nn.Module, data_loader: DataLoader, verbose=False) -> Dict[str, float]:
        logger = logging.getLogger("default")

        all_preds, all_labels = self.collect_predictions(model, data_loader, ["pred_speed", "pred_speed_regional"])

        avg_eval_res = dict() # average over all time steps
        eval_res_over_time = defaultdict(lambda: defaultdict(list))
        for seq_name in ["pred_speed", "pred_speed_regional"]:
            if verbose:
                logger.info("Evaluate model on {}".format(seq_name))

            seq_preds, seq_labels = all_preds[seq_name], all_labels[seq_name]

            # evaluate each predicted time step, i.e., forecasting from 5 min up to 1 hour
            for i in range(seq_preds.shape[1]):  # number of predicted time step
                pred = seq_preds[:, i, :]
                real = seq_labels[:, i, :]
                step_metrics = prediction_metrics(pred, real, self.ignore_value, self.mape_threshold)
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
            seq_res = prediction_metrics(seq_preds, seq_labels, self.ignore_value, self.mape_threshold)
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
            avg_eval_res[seq_name] = seq_res
            
        if self.visualize:
            # disable gradient computation, so we don't need .detach() for the model outputs
            with torch.no_grad():
                data_dict = next(iter(data_loader))
                dataset: SimBarca = data_loader.dataset
                model.eval()
                
                # debug purpose 
                # section_id_to_index = {v:k for k, v in dataset.index_to_section_id.items()}
                # for section_id in [9971, 9453, 9864, 9831, 10052]:
                #    section_num = section_id_to_index[section_id]
                section_num = torch.randint(0, dataset.node_coordinates.shape[0], (1,)).item()
                aimsun_sec_id = "_{}".format(dataset.index_to_section_id[section_num])
                dataset.visualize_batch(data_dict, model(data_dict), self.save_dir, section_num=section_num, save_note=self.save_note+aimsun_sec_id)
                dataset.plot_MAE_by_location(dataset.node_coordinates, all_preds, all_labels, save_dir=self.save_dir, save_note=self.save_note)
                dataset.plot_pred_for_section(all_preds, all_labels, self.save_dir, section_num, save_note=self.save_note+aimsun_sec_id)
            
        avg_eval_res = flatten_results_dict(avg_eval_res)
        
        # during training, the evaluation metrics will be saved to event storage, 
        # there is no need to save the average results again, but we can do it if really needed
        if self.save_res:
            save_res_to_dir(self.save_dir, avg_eval_res, self.save_note)
        
        return avg_eval_res 


def save_res_to_dir(save_dir, res, save_note="default"):
    res = dict(res)
    
    for k, v in res.items():
        if isinstance(v, dict):
            res[k] = dict(v)
        
    # save res to a json file
    with open("{}/Eval_res_{}.json".format(save_dir, save_note), "w") as f:
        json.dump(res, f, indent=4)


def invalid_to_nan(x, null_value=-1.0):
    return x.where(x != null_value, torch.nan)

def nan_to_global_avg(x):
    return torch.nan_to_num(x, nan=torch.nanmean(x))

class InputAverageModel(nn.Module):
    
    def __init__(self, seq_name):
        super().__init__()
        self.seq_name = seq_name
        
    def forward(self, data_dict):
        cluster_id = torch.as_tensor(data_dict["metadata"]["cluster_id"])
        data_seq = data_dict[self.seq_name][..., 0]
        
        pred_speed = torch.nanmean(data_seq, dim=1, keepdim=True).tile(1, 10, 1)
        regional_speed = torch.cat(
            [
                torch.nanmean(pred_speed[:, :,  cluster_id==region_id], dim=2).unsqueeze(2)
                for region_id in cluster_id.unique()
            ],
            dim=2,
        )
        return {
            "pred_speed": nan_to_global_avg(pred_speed),
            "pred_speed_regional": nan_to_global_avg(regional_speed)
        }

class LastObservedModel(nn.Module):
    
    def __init__(self, seq_name):
        super().__init__()
        self.seq_name = seq_name
    
    def forward(self, data_dict):
        cluster_id = torch.as_tensor(data_dict["metadata"]["cluster_id"])
        data_seq = data_dict[self.seq_name][..., 0]
        
        # https://github.com/pandas-dev/pandas/blob/d9cdd2ee5a58015ef6f4d15c7226110c9aab8140/pandas/core/missing.py#L224
        is_valid = torch.logical_not(torch.isnan(data_seq))
        last_valid_index = is_valid.size(1) - 1 - is_valid.flip(1).type(torch.float).argmax(dim=1)
        # generalized from this 2d indexing case (we have 3d here)
        # https://discuss.pytorch.org/t/selecting-from-a-2d-tensor-with-rows-of-column-indexes/167717/2
        # basically the indexes by torch.arange is boradcasted to match the size of last_valid_index
        # check for example data_seq[5, :, 1111] and pred_speed[5, 1111] to see if the last valid index is correct
        pred_speed = data_seq[torch.arange(data_seq.size(0)).unsqueeze(1), last_valid_index, torch.arange(data_seq.size(2)).unsqueeze(0)]
        pred_speed = pred_speed.unsqueeze(1).tile(1, 10, 1)
        pred_speed_regional = torch.cat(
            [
                torch.nanmean(pred_speed[:, :,  cluster_id==region_id], dim=2).unsqueeze(2)
                for region_id in cluster_id.unique()
            ],
            dim=2,
        )
        return {
            "pred_speed": nan_to_global_avg(pred_speed),
            "pred_speed_regional": nan_to_global_avg(pred_speed_regional)
        }

class HistoricalAverageModel(nn.Module):
    
    def __init__(self, data_loader) -> None:
        super().__init__()
        self.set_historial_avg_from_dataloader(data_loader)
    
    def set_historial_avg_from_dataloader(self, data_loader):
        """ go through the data_loader and calculate the average speed of the test set per segment
            Basically this is the best possible constant prediction
        """
        
        all_pred_speed, all_pred_speed_regional = [], []
        for batch in data_loader:
            all_pred_speed.append(batch['pred_speed'])
            all_pred_speed_regional.append(batch['pred_speed_regional'])

        all_pred_speed = torch.cat(all_pred_speed, dim=0)
        all_pred_speed_regional = torch.cat(all_pred_speed_regional, dim=0)

        N, T, P = all_pred_speed.shape
        pred_speed = torch.nanmean(all_pred_speed[:, 0, :], dim=0, keepdim=True).unsqueeze(1).tile(1, T, 1)
        regional_speed = torch.nanmean(all_pred_speed_regional[:, 0, :], dim=0, keepdim=True).unsqueeze(1).tile(1, T, 1)
        
        self.pred_speed = nan_to_global_avg(pred_speed)
        self.pred_speed_regional = nan_to_global_avg(regional_speed)

        
    def forward(self, data_dict):
        
        return {
            "pred_speed": torch.tile(self.pred_speed, (data_dict['drone_speed'].shape[0], 1, 1)),
            "pred_speed_regional": torch.tile(self.pred_speed_regional, (data_dict['drone_speed'].shape[0], 1, 1))
        }


def evaluate_trivial_models(data_loader, ignore_value=np.nan):

    logger = logging.getLogger("default")
    save_dir = "{}/{}".format("./scratch", "simbarca_trivial_baselines")
    make_dir_if_not_exist(save_dir)
    
    for model_class in [InputAverageModel, LastObservedModel]:
        for mode in ["ld_speed", "drone_speed"]:
            logger.info("Evaluating trivial models {} {}".format(model_class.__name__, mode))
            evaluator = SimBarcaEvaluator(ignore_value=ignore_value, save_dir=save_dir, save_res=True, visualize=True, save_note="{}_{}".format(model_class.__name__, mode))
            res = evaluator.evaluate(model_class(mode), data_loader, verbose=True)

    # historical average
    logger.info("Evaluating trivial models historical_avg")
    evaluator = SimBarcaEvaluator(ignore_value=ignore_value, save_dir=save_dir, save_res=True, visualize=True, save_note="HistoricalAverageModel")
    res = evaluator.evaluate(HistoricalAverageModel(data_loader=data_loader), data_loader, verbose=True)
    

if __name__ == "__main__":
    from netsanut.data.datasets import SimBarca
    from netsanut.utils.event_logger import setup_logger
    logger = setup_logger(name="default", level=logging.INFO, log_file="scratch/simbarca_trivial_baselines/eval_log.log")

    dataset = SimBarca(split="test", force_reload=False)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=dataset.collate_fn)

    evaluate_trivial_models(data_loader, ignore_value=np.nan)

    