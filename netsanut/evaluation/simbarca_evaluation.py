import json
import logging
import random

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

    def __init__(self, save_dir: str=None, save_res: bool=True) -> None:
        self.save_dir = save_dir
        self.save_res = save_res
        make_dir_if_not_exist(self.save_dir)
    
    def __call__(self, model: nn.Module, dataloader: DataLoader, **kwargs) -> Dict[str, float]:
        return self.evaluate(model, dataloader, **kwargs)

    def collect_predictions(self, 
        model: nn.Module, dataloader: DataLoader, seq_names=[]
    ) -> Dict[str, torch.Tensor]:
        """run inference of the model on the dataloader
        concatenate all predictions and corresponding labels.

        Returns: predictions, labels
        """
        model.eval()

        all_preds = defaultdict(list)
        all_labels = defaultdict(list)

        for data_dict in dataloader:
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


    def evaluate(self,
        model: nn.Module, dataloader: DataLoader, ignore_value=-1.0, mape_threshold=5, verbose=False, visualize=False, save_note="example"
    ) -> Dict[str, float]:
        logger = logging.getLogger("default")

        all_preds, all_labels = self.collect_predictions(model, dataloader, ["pred_speed", "pred_speed_regional"])

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
                step_metrics = prediction_metrics(pred, real, ignore_value=ignore_value, mape_threshold=mape_threshold)
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
            seq_res = prediction_metrics(
                seq_preds, seq_labels, ignore_value=ignore_value, mape_threshold=mape_threshold
            )
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
            
        if visualize:
            
            dataset: SimBarca = dataloader.dataset
            model.eval()
            
            section_num = torch.randint(0, dataset.node_coordinates.shape[0], (1,)).item()
            dataset.visualize_batch(data_dict, model(data_dict), self.save_dir, section_num=section_num, save_note=save_note)
            dataset.plot_MAE_by_location(dataset.node_coordinates, all_preds, all_labels, save_dir=self.save_dir, save_note=save_note)
            dataset.plot_pred_for_section(all_preds, all_labels, self.save_dir, section_num, save_note=save_note)
            
        res = flatten_results_dict(avg_eval_res)
        save_res_to_dir(self.save_dir, res, save_note)
        
        return res 


def save_res_to_dir(save_dir, res, save_note="default"):
    res = dict(res)
    make_dir_if_not_exist(save_dir)
    
    for k, v in res.items():
        if isinstance(v, dict):
            res[k] = dict(v)
        
    # save res to a json file
    with open("{}/{}.json".format(save_dir, save_note), "w") as f:
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
    
    def __init__(self, dataloader) -> None:
        super().__init__()
        self.set_historial_avg_from_dataloader(dataloader)
    
    def set_historial_avg_from_dataloader(self, dataloader):
        """ go through the dataloader and calculate the historical average speed
        """
        
        all_pred_speed, all_pred_speed_regional = [], []
        for batch in dataloader:
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


def evaluate_trivial_models(dataloader):

    save_dir = "{}/{}".format("./scratch", "simbarca_trivial_baselines")
    evaluator = SimBarcaEvaluator(save_dir, save_res=True)
    make_dir_if_not_exist(save_dir)
    
    for model_class in [InputAverageModel, LastObservedModel]:
        for mode in ["ld_speed", "drone_speed"]:
            print("Evaluating trivial models {} {}".format(model_class.__name__, mode))
            res = evaluator.evaluate(model_class(mode), dataloader, verbose=True, visualize=True, save_note="{}_{}".format(model_class.__name__, mode))
            save_res_to_dir(save_dir, res, "{}".format(mode))

    # historical average
    print("Evaluating trivial models historical_avg")
    res = evaluator.evaluate(HistoricalAverageModel(dataloader=dataloader), dataloader, verbose=True, visualize=True, save_note="HistoricalAverageModel")
    save_res_to_dir(save_dir, res, "historical_avg")
    

if __name__ == "__main__":
    from netsanut.data.datasets import SimBarca
    from netsanut.utils.event_logger import setup_logger
    logger = setup_logger(name="default", level=logging.INFO)

    dataset = SimBarca(split="test", force_reload=False)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=dataset.collate_fn)
    for data_dict in data_loader:
        print(data_dict.keys())
        break
    
    evaluate_trivial_models(data_loader)
    HA_model = HistoricalAverageModel(data_loader)
    for batch in data_loader:
        pred = HA_model(batch)
        SimBarca.visualize_batch(batch, pred, save_dir="./scratch/simbarca_trivial_baselines/")
        break
    