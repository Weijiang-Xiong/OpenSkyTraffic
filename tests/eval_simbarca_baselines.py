import json
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skytraffic.evaluation.simbarca_evaluation import SimBarcaEvaluator
from skytraffic.utils.io import make_dir_if_not_exist

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


def evaluate_trivial_models(data_loader, ignore_value=np.nan, save_dir="/scratch"):
    logger = logging.getLogger("default")
    make_dir_if_not_exist(save_dir)
    
    for model_class in [InputAverageModel, LastObservedModel]:
        for mode in ["ld_speed", "drone_speed"]:
            logger.info("Evaluating trivial models {} {}".format(model_class.__name__, mode))
            evaluator = SimBarcaEvaluator(ignore_value=ignore_value, save_dir=save_dir,visualize=True, save_note="{}_{}".format(model_class.__name__, mode))
            res = evaluator.evaluate(model_class(mode), data_loader, verbose=True)

    # historical average
    logger.info("Evaluating trivial models historical_avg")
    evaluator = SimBarcaEvaluator(ignore_value=ignore_value, save_dir=save_dir, visualize=True, save_note="HistoricalAverageModel")
    res = evaluator.evaluate(HistoricalAverageModel(data_loader=data_loader), data_loader, verbose=True)
    

if __name__ == "__main__":
    from skytraffic.data.datasets import SimBarcaMSMT
    from skytraffic.utils.event_logger import setup_logger
    make_dir_if_not_exist("scratch/simbarca_trivial_baselines")
    logger = setup_logger(name="default", level=logging.INFO, log_file="scratch/simbarca_trivial_baselines/eval_log.log")

    dataset = SimBarcaMSMT(split="test")
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=dataset.collate_fn)
    evaluate_trivial_models(data_loader, ignore_value=np.nan, save_dir="scratch/simbarca_trivial_baselines")
    
    from skytraffic.data.datasets import SimBarcaRandomObservation
    dataset = SimBarcaRandomObservation(split="test")
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=dataset.collate_fn)
    evaluate_trivial_models(data_loader, ignore_value=np.nan, save_dir="scratch/simbarcarnd_trivial_baselines")

    