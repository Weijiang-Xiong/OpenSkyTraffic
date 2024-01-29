import json
import tqdm
import logging

from pathlib import Path
from collections import defaultdict
from typing import List, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import netsanut
from netsanut.data import DATASET_CATALOG
from netsanut.evaluation.metrics import prediction_metrics
from netsanut.util import flatten_results_dict

_package_init_file = netsanut.__file__
_root: Path = (Path(_package_init_file).parent.parent).resolve()
assert _root.exists(), "please check detectron2 installation"


class SimBarca(Dataset):
    data_root = "datasets/simbarca"
    meta_data_folder = "datasets/simbarca/metadata"
    input_seqs = ["drone_speed", "ld_speed"]
    output_seqs = ["pred_speed", "pred_speed_regional"]
    train_split_size = 0.75

    def __init__(self, split="train", force_reload=False):
        self.split = split
        self.force_reload = force_reload
        self.read_graph_structure()
        samples = self.load_or_process_samples()
        for attribute in self.sequence_names:
            setattr(
                self,
                attribute,
                torch.as_tensor(samples[attribute], dtype=torch.float32),
            )
        # self.drone_speed = samples['drone_speed']
        # self.ld_speed = samples['ld_speed']
        # self.pred_speed = samples['pred_speed']
        # self.pred_speed_regional = samples['pred_speed_regional']
        self.metadata = self.set_metadata_dict()
        self.data_augmentations = []

    def __getitem__(self, index):
        # don't use tuple comprehension here, otherwise it will return a generator instead of actual data
        data_seqs = [getattr(self, attribute)[index] for attribute in self.sequence_names]

        return data_seqs

        # return self.drone_speed[index], self.ld_speed[index], self.pred_speed[index], self.pred_speed_regional[index]

    def __len__(self):
        return self.drone_speed.shape[0]

    @property
    def sequence_names(self):
        return self.input_seqs + self.output_seqs

    def get_processed_file_name(self):
        return "{}/processed/{}.npz".format(self.data_root, self.split)

    def get_split_samples(self, sample_files: List[str]):
        # a sample file contains all training samples from a simulation session
        train_test_split_index = int(self.train_split_size * len(sample_files))

        match self.split:
            case "train":
                return sample_files[:train_test_split_index]
            case "test":
                return sample_files[train_test_split_index:]
            case _:
                raise ValueError("split {} not supported".format(self.split))

    def set_metadata_dict(self):
        metadata = {
            "adjacency": self.adjacency,
            "cluster_id": self.cluster_id,
            "grid_id": self.grid_id,
        }

        metadata["mean_and_std"] = {
            att: (
                torch.mean(getattr(self, att)[..., 0]),
                torch.std(getattr(self, att)[..., 0]),
            )
            for att in self.sequence_names
        }
        metadata["input_seqs"] = self.input_seqs
        metadata["output_seqs"] = self.output_seqs

        return metadata

    def read_graph_structure(self):
        folder = "{}/{}".format(_root, self.meta_data_folder)
        connections = pd.read_csv(
            "{}/connections.csv".format(folder),
            dtype={
                "turn": int,
                "org": int,
                "dst": int,
                "intersection": int,
                "length": float,
            },
        )
        link_bboxes = pd.read_csv(
            "{}/link_bboxes_clustered.csv".format(folder),
            dtype={
                "id": int,
                "from_x": float,
                "from_y": float,
                "to_x": float,
                "to_y": float,
                "length": float,
                "out_ang": float,
                "num_lanes": int,
            },
        )
        with open("{}/intersec_polygon.json".format(folder), "r") as f:
            intersection_polygon = json.load(f)

        link_bboxes = link_bboxes.sort_values(by=["id"])
        section_ids_sorted = link_bboxes["id"].to_numpy()
        section_id_to_index = {link_id: index for index, link_id in enumerate(section_ids_sorted)}
        index_to_section_id = {index: section_id for index, section_id in enumerate(section_ids_sorted)}
        section_cluster = link_bboxes["cluster"].to_numpy()  # check with the csv file
        section_grid_id = link_bboxes["grid_nb"].to_numpy()  # check with the csv file
        node_coordinates = link_bboxes[["c_x", "c_y"]].to_numpy()

        adjacency_matrix = np.zeros((len(section_ids_sorted), len(section_ids_sorted)))
        for row in connections.itertuples():
            adjacency_matrix[section_id_to_index[row.org], section_id_to_index[row.dst]] = 1

        self.adjacency = adjacency_matrix
        self.node_coordinates = node_coordinates
        self.cluster_id: np.ndarray = section_cluster
        self.grid_id: np.ndarray = section_grid_id
        self.index_to_section_id: Dict = index_to_section_id

    def load_or_process_samples(self) -> List[Dict[str, pd.DataFrame | np.ndarray]]:
        if Path(self.get_processed_file_name()).exists() and not self.force_reload:
            print("Trying to load existing processed samples for {} split".format(self.split))
            with open(self.get_processed_file_name(), "rb") as f:
                loaded_data = np.load(f)
                return {key: value for key, value in loaded_data.items()}

        print("No processed samples found or forced to reload, processing samples from scratch")
        all_sample_data = defaultdict(list)
        sample_files = sorted(
            list(Path("{}/{}".format(_root, self.data_root)).glob("session_*/timeseries/samples.npz"))
        )
        split_sample_files = self.get_split_samples(sample_files)

        print("Found {} samples for {} split, reading them one by one".format(len(split_sample_files), self.split))
        for sample_file in tqdm.tqdm(split_sample_files):
            with open(sample_file, "rb") as f:
                sample_data: np.lib.npyio.NpzFile = np.load(f)
                for key in sample_data:
                    all_sample_data[key].append(sample_data[key])

        # concatenate the samples along the batch dimension
        for key, value in all_sample_data.items():
            all_sample_data[key] = np.concatenate(value, axis=0)

        print("Processing per section data")
        # process the input and predicted speed
        processed_samples = dict()
        # input and predict speed per section, using v = total_dist / total_time
        # each array in all_sample_data have shape (N, T, P, 2), where N is the number of samples, T is the number of time steps, P is the number of sections, and the last dimension 2 corresponds to (time_in_day, value)
        for mod_type in ["drone", "pred"]:
            vdist_values: np.ndarray = all_sample_data["{}_vdist".format(mod_type)][
                ..., 0
            ]  # total vehicle distance
            vdist_tind: np.ndarray = all_sample_data["{}_vdist".format(mod_type)][..., 1]  # time in day
            vtime_values: np.ndarray = all_sample_data["{}_vtime".format(mod_type)][..., 0]  # total vehicle time
            # replace nan values with -1, because it means both distance and time are 0, no vehicle detected
            # this is different from zero speed, which can also happen when all vehicles are stopped at red lights
            speed_values = np.nan_to_num(vdist_values / vtime_values, nan=-1)
            processed_samples["{}_speed".format(mod_type)] = np.stack([speed_values, vdist_tind], axis=-1)
        processed_samples["ld_speed"] = all_sample_data["ld_speed"]

        print("Processing regional data")
        regional_speed = []
        # aggregate link into regions
        for region_id in np.unique(self.cluster_id):
            region_mask = self.cluster_id == region_id
            region_vdist_values = np.mean(all_sample_data["pred_vdist"][..., region_mask, 0], axis=-1)
            # add the time in day here, the first index
            # time in day was copied for all positions, so taking 1 is enough
            region_vdist_tind = all_sample_data["pred_vdist"][..., region_mask, 1][..., 0]
            region_vtime_values = np.mean(all_sample_data["pred_vtime"][..., region_mask, 0], axis=-1)
            region_speed_values = np.nan_to_num(region_vdist_values / region_vtime_values, nan=-1)
            regional_speed.append(np.stack([region_speed_values, region_vdist_tind], axis=-1))
        # the elements have shape (N, T, 2), where 2 corresponds to (time_in_day, value)
        # we stack them into shape (N, T, R, 2) where R is the number of regions
        processed_samples["pred_speed_regional"] = np.stack(regional_speed, axis=2)

        # save all samples as a compressed npz file
        print("Saving processed samples to {}".format(self.get_processed_file_name()))
        with open(self.get_processed_file_name(), "wb") as f:
            np.savez_compressed(f, **processed_samples)

        return processed_samples

    def collate_fn(self, list_of_seq: List[Dict]) -> Dict[str, torch.Tensor]:
        data_dict = dict()
        for attr_id, attribute in enumerate(self.sequence_names):
            data_dict[attribute] = torch.cat(
                [seq[attr_id].unsqueeze(0) for seq in list_of_seq], dim=0
            ).contiguous()

        # assume the output values have 0 index in the last dimension,
        # the other dimensions are time in day, day in week etc.
        for name in self.output_seqs:
            data_dict[name] = data_dict[name][..., 0]

        for aug in self.data_augmentations:
            data_dict = aug(data_dict)

        data_dict["metadata"] = self.metadata

        return data_dict


def register_simbarca():
    pass


def build_train_loader():
    dataset = SimBarca(split="train", force_reload=False)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=dataset.collate_fn)

    return data_loader


def build_test_loader():
    dataset = SimBarca(split="test", force_reload=False)
    data_loader = DataLoader(dataset, batch_size=8, collate_fn=dataset.collate_fn)

    return data_loader


def build_trainvaltest_loaders():
    dataloaders = {
        "train": build_train_loader(),
        "val": build_test_loader(),
        "test": build_test_loader(),
    }

    metadata = dataloaders["train"].dataset.metadata

    return dataloaders, metadata


def inference_on_dataset(model: nn.Module, dataloader: DataLoader, seq_names=[]) -> Dict[str, torch.Tensor]:
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

        for name in seq_names:
            all_preds[name].append(pred_dict[name])
            all_labels[name].append(data_dict[name])

    for res_collection in [all_preds, all_labels]:
        for key, value in res_collection.items():
            res_collection[key] = torch.cat(value, dim=0).detach().cpu()

    return all_preds, all_labels


def evaluate(
    model: nn.Module, dataloader: DataLoader, verbose=False, ignore_value=-1.0, mape_threshold=5
) -> Dict[str, float]:
    logger = logging.getLogger("default")

    all_preds, all_labels = inference_on_dataset(model, dataloader, ["pred_speed", "pred_speed_regional"])

    all_eval_res = dict()
    for seq_name in ["pred_speed", "pred_speed_regional"]:
        if verbose:
            logger.info("Evaluate model on {}".format(seq_name))

        seq_preds, seq_labels = all_preds[seq_name], all_labels[seq_name]

        # evaluate each predicted time step, i.e., forecasting from 5 min up to 1 hour
        for i in range(seq_preds.shape[1]):  # number of predicted time step
            pred = seq_preds[:, i, :]
            real = seq_labels[:, i, :]
            step_metrics = prediction_metrics(pred, real, ignore_value=ignore_value, mape_threshold=mape_threshold)

            if verbose:
                logger.info("Evaluate model on test data at {:d} time step".format(i + 1))
                logger.info(
                    "MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}".format(
                        step_metrics["mae"], step_metrics["mape"], step_metrics["rmse"]
                    )
                )

        # average performance on all prediction steps, usually not reported in papers
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

        all_eval_res[seq_name] = seq_res

    return flatten_results_dict(all_eval_res)

class TrivialModel(nn.Module):
    
    def __init__(self, seq_name, fun_type="avg"):
        super().__init__()
        self.seq_name = seq_name
        self.func_type = fun_type
        
    def forward(self, data_dict):
        cluster_id = torch.as_tensor(data_dict["metadata"]["cluster_id"])
        match self.func_type:
            case "avg":
                pred_speed = torch.mean(data_dict[self.seq_name][..., 0], dim=1, keepdim=True).tile(1, 10, 1)
                regional_speed = torch.cat(
                    [
                        torch.mean(pred_speed[:, :,  cluster_id==region_id], dim=2).unsqueeze(2)
                        for region_id in cluster_id.unique()
                    ],
                    dim=2,
                )
                return {
                    "pred_speed": pred_speed,
                    "pred_speed_regional": regional_speed
                }
            case "last":
                pred_speed = data_dict[self.seq_name][:, -1, :, 0].unsqueeze(1).tile(1, 10, 1)
                regional_speed = torch.cat(
                    [
                        torch.mean(pred_speed[:, :,  cluster_id==region_id], dim=2).unsqueeze(2)
                        for region_id in cluster_id.unique()
                    ],
                    dim=2,
                )
                return {
                    "pred_speed": pred_speed,
                    "pred_speed_regional": regional_speed
                }
        return 



def evaluate_trivial_models(dataloader):
    
    print("Evaluating trivial models ld_speed AVG")
    print(evaluate(TrivialModel("ld_speed", fun_type="avg"), dataloader, verbose=True))
    print("Evaluating trivial models ld_speed LAST")
    print(evaluate(TrivialModel("ld_speed", fun_type="last"), dataloader, verbose=True))
    print("Evaluating trivial models drone_speed AVG")
    print(evaluate(TrivialModel("drone_speed", fun_type="avg"), dataloader, verbose=True))
    print("Evaluating trivial models drone_speed LAST")
    print(evaluate(TrivialModel("drone_speed", fun_type="last"), dataloader, verbose=True))


if __name__.endswith(".simbarca"):
    """this happens when something is imported from this file
    we can register the dataset here
    """
    register_simbarca()

if __name__ == "__main__":
    train_loader = build_train_loader()
    for data_dict in train_loader:
        print(data_dict.keys())
        break

    test_loader = build_test_loader()
    for data_dict in test_loader:
        print(data_dict.keys())
        break

    evaluate_trivial_models(test_loader)
