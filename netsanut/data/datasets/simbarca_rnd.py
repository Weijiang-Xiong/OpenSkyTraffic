import os
import pickle
import logging
import argparse
from typing import Dict

import numpy as np
import seaborn as sns 

import torch

from ..catalog import DATASET_CATALOG
from .simbarca import SimBarca

sns.set_style("darkgrid")
logger = logging.getLogger("default")

def add_gaussian_noise(generator:torch.Generator, data:torch.Tensor, std=0.1):
    noise = torch.normal(0, std, size=data.size()[:-1], generator=generator)
    data[..., 0] = data[..., 0] + noise * data[..., 0]
    return data

class SimBarcaRandomObservation(SimBarca):
    """ This class implements randomized drone observations and loop detector observations. The purpose is to make the dataset more realistic, because in reality, we don't have loop detectors for all road segments, and we can hardly fly enough drones to cover a whole city. Therefore, dealing with partial information is inevitable. 

    Concretely, 10% of the road segments will have loop detector observations, which will be initialized at the first time and later saved to file for reuse. In this way the loop detector positions are fixed across experiments, and the results can be fairly compared. 
    
    The drone observations will be available for random 10% of the grid IDs (but not directly road segments), since we assume a drone to observe nearly all vehicles in its square-shaped FOV. To imitate a real-world scenario, each simulation session will be regarded as an individual day, when we fly the drones in different ways. We want the drone positions to mimic the flight plan for the simulation sessions, which will be consistent in all epochs in the same experiment and also across all experiments.
    
    While it's simple that the training and testing splits should share the same loop detector positions, the drone positions are more complicated. The training samples are generated in a sliding-window way based on the traffic statistics of the whole simulation. Therefore the drone positions should also be a "sliding-window", which means neighboring samples should have overlapped drone positions with only 1 time step difference. Check `load_or_init_drone_pos` for implementation. 
    
    Besides, for the training split, we exclude the data of the unmonitored road segments from both the input and output, which means partial input and partial label for the model to learn. The labels for regional speed predictions will also be created based on partial information (which makes it biased but that's the best we could do for now). However, for the test set, we still use the full information for evaluation (so don't train on the test set).
    """
    
    # we need them to recompute the regional speed values with the monitoring mask
    aux_seqs = ["pred_vdist", "pred_vtime"]

    def __init__(
        self,
        ld_cvg=0.1,
        drone_cvg=0.1,
        reinit_pos=False,
        mask_seed=42,
        use_clean_data=True,
        noise_seed=114514,
        drone_noise=0.05,
        ld_noise=0.15,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pred_vdist: torch.Tensor
        self.pred_vtime: torch.Tensor
        self.ld_cvg = ld_cvg
        self.drone_cvg = drone_cvg
        self.drone_noise = drone_noise
        self.ld_noise = ld_noise
        self.mask_seed = mask_seed
        self.reinit_pos = reinit_pos
        self.use_clean_data = use_clean_data
        self.noise_seed = noise_seed
        
        self.ld_mask: torch.Tensor
        self.drone_flight_mask: torch.Tensor
        self.load_or_init_ld_mask()
        self.load_or_init_drone_mask()
        
        if self.split == "train":
            self.add_seqs = ['ld_mask', 'drone_mask', 'pred_mask']
        elif self.split == "test":
            # output sequences are not masked in the test set, to test on the full information
            self.add_seqs = ['ld_mask', 'drone_mask']
        
        # these sequences are saved ahead of time, so the noises can be added directly
        # but the regional speed values are recomputed using the available segment-level data, so it's put to 
        # the function `apply_masking`
        if not use_clean_data:
            
            logger.info("Using random seed {} for adding noise to the data".format(self.noise_seed))
            self.rnd_generator = torch.Generator()
            self.rnd_generator.manual_seed(self.noise_seed)
            
            logger.info("Using corrupted data for train set, but clean label for test set")
            # NOTE quick and dirty data augmentation ... not configuratle, not scalable ... 
            if self.split == "train":
                logger.info("Adding Gaussian noise to the training input and label")
                # we train on corrupted data to make the model more robust
                self.drone_speed = add_gaussian_noise(self.rnd_generator, self.drone_speed, std=drone_noise)
                self.ld_speed = add_gaussian_noise(self.rnd_generator, self.ld_speed, std=ld_noise)
                self.pred_speed = add_gaussian_noise(self.rnd_generator, self.pred_speed, std=drone_noise)
                # add noises to the vehicle distance and time for regional speed computation
                self.pred_vdist = add_gaussian_noise(self.rnd_generator, self.pred_vdist, std=drone_noise)
                self.pred_vtime = add_gaussian_noise(self.rnd_generator, self.pred_vtime, std=drone_noise)
            elif self.split == "test":
                # for test data, the input is corrupted but the label is not for evaulating the model
                logger.info("Adding Gaussian noise to the testing input (BUT NOT lable)")
                self.drone_speed = add_gaussian_noise(self.rnd_generator, self.drone_speed, std=drone_noise)
                self.ld_speed = add_gaussian_noise(self.rnd_generator, self.ld_speed, std=ld_noise)
    
    @property
    def ld_mask_file(self):
        return "{}/processed/ld_mask_{}.pkl".format(self.data_root, int(self.ld_cvg * 100))

    @property
    def drone_mask_file(self):
        return "{}/processed/drone_mask_{}_{}.pkl".format(self.data_root, int(self.drone_cvg*100), self.split)
    
    def load_or_init_ld_mask(self):
        if os.path.exists(self.ld_mask_file) and not self.reinit_pos:
            with open(self.ld_mask_file, "rb") as f:
                logger.info("Loading loop detector mask from file")
                ld_mask = pickle.load(f)
            if ld_mask.sum() != int(self.ld_cvg * self.adjacency.shape[0]):
                logger.warning("The loop detector coverage in the file are not consistent with the current settings, check `ld_cvg` argument or set `reinit_pos=True`")
        else:
            logger.info("Initializing loop detector mask using random seed {} and coverage {}".format(self.mask_seed, self.ld_cvg))
            rng = np.random.default_rng(self.mask_seed)
            # exclude the locations with too many nan, (which means these roads have insufficient vehicles)
            # and then randomly select a few indexes of the road segments to have loop detectors
            nan_by_location = np.mean(torch.isnan(self.ld_speed[..., 0]).numpy().astype(float), axis=(0, 1))
            # maybe it's OK to have 10% nan values, then we have ~92% valid loop detectors
            valid_pos = np.nonzero(nan_by_location < 0.1)[0] 
            if self.ld_cvg >= 1.0:
                ld_mask = np.ones(shape=self.adjacency.shape[0], dtype=bool)
            # still require more loop detectors than the valid positions (92%~99%, but unlikely)
            elif self.ld_cvg >= len(valid_pos) / len(nan_by_location):
                valid_pos = np.argsort(nan_by_location)[:int(self.ld_cvg * self.adjacency.shape[0])]
            else:
                ld_pos = rng.choice(valid_pos, size=int(self.ld_cvg * self.adjacency.shape[0]), replace=False)
                ld_mask = np.zeros(shape=self.adjacency.shape[0], dtype=bool)
                ld_mask[ld_pos] = True
            
            with open(self.ld_mask_file, "wb") as f:
                pickle.dump(ld_mask, f)

        # the dtype must be bool, otherwise torch will regard it as indexes of the items to be selected
        self.ld_mask = torch.as_tensor(ld_mask, dtype=torch.bool)

    def load_or_init_drone_mask(self):
        
        if os.path.exists(self.drone_mask_file) and not self.reinit_pos:
            logger.info("Loading drone mask from file")
            all_drone_mask = pickle.load(open(self.drone_mask_file, "rb"))
            if np.abs(all_drone_mask.mean() - self.drone_cvg) > 0.05:
                logger.warning("The drone coverage in the file appears to be higher than config, check if `drone_cvg` argument has changed or set `reinit_pos=True`")
        else:
            logger.info("Initializing drone mask using random seed {} and coverage {}".format(self.mask_seed, self.drone_cvg))
            grid_cells = np.sort(np.unique(self.grid_id))
            rng = np.random.default_rng(self.mask_seed + 777) # avoid using the same random seed as ld_pos
            
            all_drone_mask = []
            for _ in range(len(self) // self.sample_per_session): # different simulation session
                # init drone positions for the first sample in each session
                # 30min input, 30min output, change every 3 mins, so that's 20 x num_grid 
                drone_mask = np.stack(
                    [np.isin(self.grid_id, 
                             rng.choice(grid_cells, size=int(self.drone_cvg * len(grid_cells)), replace=False)
                    ) for _ in range(20)]
                )
                all_drone_mask.append(drone_mask)
                # for every 3 min, sample a new set of drone positions and discard the earliest step
                for _ in range(1, self.sample_per_session):
                    next_step_drone_mask = np.isin(
                        self.grid_id, 
                        rng.choice(grid_cells, size=(int(self.drone_cvg * len(grid_cells))), replace=False)
                    )
                    drone_mask = np.concatenate((drone_mask[1:, :], next_step_drone_mask.reshape(1, -1)), axis=0)
                    all_drone_mask.append(drone_mask)
            all_drone_mask = np.stack(all_drone_mask, axis=0)
            with open(self.drone_mask_file, "wb") as f:
                pickle.dump(all_drone_mask, f)
        
        # the dtype must be bool, otherwise torch will regard it as indexes of the items to be selected
        self.drone_flight_mask = torch.as_tensor(all_drone_mask, dtype=torch.bool)

    def apply_masking(self, sample: Dict, ld_mask:torch.Tensor, drone_flight_mask: torch.Tensor) -> Dict:
        """ Apply the masking of loop detector and drone data to the input and label
        
            For both train and test sets:
                1. Set all INPUT modalities for unmonitored road segments to nan
            For train set only:
                1. Set all OUTPUT modalities for unmonitored road segments to nan
                2. Recalculate regional speed values based on the monitored road segments

            For the test set, we keep the output values as they are, because we want to evaluate the model's performance on the full information, even when the model is trained with partial data.
        """
        sample['ld_speed'][:, ld_mask == 0, 0] = torch.nan
        sample['ld_mask'] = ld_mask.bool()
        # the input and output sequences corresponds to a continuous 1 hour time window
        # and the drone_flight_mask is for this 1 hour. We take the first half hour for input 
        drone_mask = drone_flight_mask[:int(drone_flight_mask.shape[0]/2), :]
        # drone speeds are given every 5 seconds, but drones are assumed to change positions every 3 minutes
        # so we need to repeat the mask to match the time resolution
        drone_mask = torch.repeat_interleave(
                            drone_mask, 
                            int(sample['drone_speed'].shape[0]/drone_mask.shape[0]), 
                            dim=0)
        sample['drone_speed'][..., 0][drone_mask == 0] = torch.nan
        sample['drone_mask'] = drone_mask.bool()
        
        if self.split == "train":
            pred_mask = drone_flight_mask[int(drone_flight_mask.shape[0]/2):, :]
            sample['pred_speed'][..., 0][pred_mask == 0] = torch.nan
            sample['pred_mask'] = pred_mask
            
            # compute regional speed based on the monitored road segments
            regional_speed = []
            # aggregate link into regions
            for region_id in torch.unique(self.cluster_id):
                region_mask = torch.logical_and((self.cluster_id == region_id).reshape(1, -1), pred_mask)
                # sum the total distance but ignore NaN values, the sum will be NaN if one element is NaN
                region_vdist_values = torch.nansum(sample["pred_vdist"][..., 0]*region_mask.float(), dim=-1) 
                # add the time in day here, the first index
                # time in day was copied for all positions, so taking 1 is enough
                region_vdist_tind = sample["pred_vdist"][..., 0, 1]
                region_vtime_values = torch.nansum(sample["pred_vtime"][..., 0]*region_mask.float(), dim=-1)
                region_speed_values = region_vdist_values / region_vtime_values
                regional_speed.append(torch.stack([region_speed_values, region_vdist_tind], dim=-1))
            # the elements have shape (N, T, 2), where 2 corresponds to (time_in_day, value)
            # we stack them into shape (N, T, R, 2) where R is the number of regions
            sample["pred_speed_regional"] = torch.stack(regional_speed, dim=1)
        
        return sample
        
    def __getitem__(self, index):
        """ 
        In the files, we save the drone mask every 3 mins, since we assume the drones will move to monitor different locations every 3 minutes. However, the drone input is given every 5 seconds, so we need to extend the drone mask to the same time resolution as the drone input.
        
        The mask for loop detector is a 0-1 tensor of shape P, where P is the number of road segments. Since loop detectors are installed as fixed infrastructure, ld_mask remain unchanged over time.
        The mask for drone input of ONE sample is also a 0-1 tensor but it has shape (T, P), where P is the number of road segments, and T is the number of time steps. To avoid complex transition states, we assume drones to jump from one grid to another. 
        """
        data_dict = super().__getitem__(index)
        data_dict = self.apply_masking(data_dict, self.ld_mask, self.drone_flight_mask[index])
        
        # after applying the mask, we don't need the vehicle distance and time for training or testing
        del data_dict['pred_vdist']
        del data_dict['pred_vtime']
        
        return data_dict
    

if __name__.endswith(".simbarca_rnd"):
    """this happens when something is imported from this file
    we can register the dataset here
    """
    DATASET_CATALOG['simbarca_rnd_train'] = lambda **args: SimBarcaRandomObservation(split='train', **args)
    DATASET_CATALOG['simbarca_rnd_test'] = lambda **args: SimBarcaRandomObservation(split='test', **args)
    
if __name__ == "__main__":
    
    from netsanut.utils.event_logger import setup_logger
    logger = setup_logger(name="default", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-scratch", action="store_true", help="Process everything from scratch")
    args = parser.parse_args()

    debug_set = SimBarcaRandomObservation(split='train', reinit_pos=args.from_scratch, use_clean_data=False)
    sample = debug_set[0]
    batch = debug_set.collate_fn([debug_set[0], debug_set[100]])
    
    debug_set = SimBarcaRandomObservation(split='test', reinit_pos=args.from_scratch, use_clean_data=False)
    sample = debug_set[0]
    batch = debug_set.collate_fn([debug_set[0], debug_set[100]])
