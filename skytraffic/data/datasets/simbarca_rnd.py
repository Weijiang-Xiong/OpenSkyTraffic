import os
import pickle
import logging
from typing import Dict, List

import numpy as np

import torch

from .simbarca_msmt import SimBarcaMSMT

logger = logging.getLogger("default")

def add_gaussian_noise(generator:torch.Generator, data:torch.Tensor, std=0.1):
    noise = torch.normal(0, std, size=data.size(), generator=generator)
    data = data + noise * data
    return data

class SimBarcaRandomObservation(SimBarcaMSMT):
    """ This class implements randomized drone observations and loop detector observations. The purpose is to make the dataset more realistic, because in reality, we don't have loop detectors for all road segments, and we can hardly fly enough drones to cover a whole city. Therefore, dealing with partial information is inevitable. 

    Concretely, 10% of the road segments will have loop detector observations, which will be initialized at the first time and later saved to file for reuse. In this way the loop detector positions are fixed across experiments, and the results can be fairly compared. 
    
    The drone observations will be available for random 10% of the grid IDs (but not directly road segments), since we assume a drone to observe nearly all vehicles in its square-shaped FOV. To imitate a real-world scenario, each simulation session will be regarded as an individual day, when we fly the drones in different ways. We want the drone positions to mimic the flight plan for the simulation sessions, which will be consistent in all epochs in the same experiment and also across all experiments.
    
    While it's simple that the training and testing splits should share the same loop detector positions, the drone positions are more complicated.
    We assume a drone to relocate every 3 minutes, and this happens instantly, so the drones are literally jumping from one grid to another. 
    
    For train set, we use partially observed vehicle travel distance and time to compute segment and regional speed as training data, this is to imitate a real-world scenario where we are only able to monitor part of the area.
    However, for test set, we use the clean and full data, to evaluate the model's ability to infer the complete traffic state even if it is given only partial data.
    """
    def __init__(self, split="train", ld_cvg=0.1, drone_cvg=0.1, reinit_pos=False, mask_seed=42, use_clean_data=False, noise_seed=114514, drone_noise=0.05, ld_noise=0.15, **kwargs):
        # Set attributes before calling super().__init__() because the parent constructor
        # calls prepare_data_for_prediction() which needs these attributes
        self.ld_cvg = ld_cvg
        self.drone_cvg = drone_cvg
        self.reinit_pos = reinit_pos
        self.mask_seed = mask_seed
        self.use_clean_data = use_clean_data
        self.noise_seed = noise_seed
        self.drone_noise = drone_noise
        self.ld_noise = ld_noise

        # in addition to the data sequences in the parent class (e.g., drone_speed, ld_speed)
        # this random observation case implements the masking of loop detector and drones
        self.ld_mask: torch.Tensor # shape (P,)
        # shape (S, T_low, G) the drones are relocated at a lower time resolution, and G is the number of grid cells
        self.drone_flight_mask: torch.Tensor
        self._r = 36 # the ratio of 3min to 5s, we need this to expand the size of drone masks to cope with two different time resolutions

        # these are additional sequences in the input batch, they are not normalized like those in io_seqs
        if split == "train":
            self.add_seqs = ['ld_mask', 'drone_mask', 'pred_mask']
        elif split == "test":
            # output sequences are not masked in the test set, to test on the full information
            self.add_seqs = ['ld_mask', 'drone_mask']

        # Note: the parent class will call these functions sequentially, so we don't need to call them here
        # 1. self.prepare_data_for_prediction()
        # 2. self.load_or_compute_metadata()
        # 3. self.clean_up_raw_sequences()
        super().__init__(split=split, **kwargs)


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
                ld_mask = np.zeros(shape=self.adjacency.shape[0], dtype=bool)
                ld_mask[valid_pos] = True
            else:
                ld_pos = rng.choice(valid_pos, size=int(self.ld_cvg * self.adjacency.shape[0]), replace=False)
                ld_mask = np.zeros(shape=self.adjacency.shape[0], dtype=bool)
                ld_mask[ld_pos] = True
            
            with open(self.ld_mask_file, "wb") as f:
                pickle.dump(ld_mask, f)

        # the dtype must be bool, otherwise torch will regard it as indexes of the items to be selected
        self.ld_mask = torch.as_tensor(ld_mask, dtype=torch.bool)

    def load_or_init_drone_flight_mask(self, mask_length:int):
        
        if os.path.exists(self.drone_mask_file) and not self.reinit_pos:
            logger.info("Loading drone mask from file")
            drone_flight_masks = pickle.load(open(self.drone_mask_file, "rb"))
            if np.abs(drone_flight_masks.mean() - self.drone_cvg) > 0.05:
                logger.warning("The drone coverage in the file appears to be higher than config, check if `drone_cvg` argument has changed or set `reinit_pos=True`")
        else:
            logger.info("Initializing drone mask using random seed {} and coverage {}%".format(self.mask_seed, int(self.drone_cvg*100)))
            grid_cells = np.sort(np.unique(self.grid_id))
            rng = np.random.default_rng(self.mask_seed + 777) # avoid using the same random seed as ld_pos
            
            drone_flight_masks = []
            for _ in range(self.num_sessions): # different simulation session
                # init drone positions for the first sample in each session
                drone_mask = np.stack(
                    [np.isin(self.grid_id, 
                             rng.choice(grid_cells, size=int(self.drone_cvg * len(grid_cells)), replace=False)
                    ) for _ in range(mask_length)]
                )
                drone_flight_masks.append(drone_mask)
            drone_flight_masks = np.stack(drone_flight_masks, axis=0)

            with open(self.drone_mask_file, "wb") as f:
                pickle.dump(drone_flight_masks, f)
        
        # the dtype must be bool, otherwise torch will regard it as indexes of the items to be selected
        self.drone_flight_mask = torch.as_tensor(drone_flight_masks, dtype=torch.bool)

    def prepare_data_for_prediction(self):
        """
        The data sequence preparation has the following steps:
            1. Prepare time-in-day encoding
            2. Prepare clean data for evaluation (before masking)
            3. Copy the data sequences for modification purpose
            4. Prepare corrupted data for training (with copied sequences)
            5. Load and apply loop detector mask and drone flight masks
            6. Calculate segment speed and regional speed based on partial and noisy observation

        We assume partial and noisy observation for the training set and the input of test set, but we evaluate the model's performance on the clean and complete data.
        The purpose is to evaluate the model's performance to infer the complete traffic state even if it is given only partial data.
        """
        self.prepare_time_steps()

        # copy the data sequences for modification purpose
        ld_speed_3min = torch.as_tensor(self._ld_speed_3min, dtype=torch.float32).clone()
        vdist_3min = torch.as_tensor(self._vdist_3min, dtype=torch.float32).clone()
        vtime_3min = torch.as_tensor(self._vtime_3min, dtype=torch.float32).clone()
        vdist_5s = torch.as_tensor(self._vdist_5s, dtype=torch.float32).clone()
        vtime_5s = torch.as_tensor(self._vtime_5s, dtype=torch.float32).clone()

        # load and apply loop detector mask and drone flight masks
        self.load_or_init_ld_mask()
        # we assume the drones to change location every 3 minutes, and this happens instantly
        self.load_or_init_drone_flight_mask(mask_length=len(self.time_in_day_3min))

        # set the unmonitored road segments to nan
        ld_speed_3min[..., self.ld_mask == 0] = torch.nan
        vdist_3min[self.drone_flight_mask == 0] = torch.nan
        vtime_3min[self.drone_flight_mask == 0] = torch.nan

        # expand the drone mask to the 5s time resolution, shape (S, T_low, P) -> (S, T_high, P)
        # this is because the drone input is given every 5s, but the drone mask is given every 3min
        # so we need to repeat the mask to match the time resolution
        
        mask = torch.repeat_interleave(self.drone_flight_mask, self._r, dim=1)
        # pad the time dimension with zeros to make the size of mask the same as vdist_5s
        mask = torch.cat([mask, torch.zeros_like(vdist_5s)[:, mask.shape[1]:vdist_5s.shape[1], :]], dim=1)

        vdist_5s[mask == 0] = torch.nan
        vtime_5s[mask == 0] = torch.nan

        self.ld_speed = ld_speed_3min
        self.drone_speed = vdist_5s / vtime_5s

        # we need to add noise to the input for both training and testing
        if not self.use_clean_data:
            logger.info("Adding noise to input speed data using random seed {}".format(self.noise_seed))
            self.rnd_generator = torch.Generator()
            self.rnd_generator.manual_seed(self.noise_seed)
            self.drone_speed = torch.clamp(
                add_gaussian_noise(self.rnd_generator, self.drone_speed, std=self.drone_noise), 
                min=0, max=14.0
                )
            self.ld_speed = torch.clamp(
                add_gaussian_noise(self.rnd_generator, self.ld_speed, std=self.ld_noise), 
                min=0, max=14.0
                )

        # for labels, we only add noise to the training set, and use the clean data for evaluation
        # this is to evaluate the model's ability to infer the complete traffic state even if it is given only
        # partial and noisy data
        if self.split == "train":
            logger.info("Using partially observable data for training")
            pred_speed = vdist_3min / vtime_3min
            # clip the speed to be positive and less than 14 m/s
            pred_speed_regional = torch.as_tensor(
                self.regional_speed_from_segment_numpy(vdist_3min.numpy(), vtime_3min.numpy()), 
                dtype=torch.float32
                )
            # Noisy data, but we add a sanity check that the speed can not be negative and can't exceed the limit
            if not self.use_clean_data:
                logger.info("Adding noise to labels using random seed {}".format(self.noise_seed))
                pred_speed = torch.clamp(
                    add_gaussian_noise(self.rnd_generator, pred_speed, std=self.drone_noise), 
                    min=0, max=14.0
                    )
                pred_speed_regional = torch.clamp(
                    add_gaussian_noise(self.rnd_generator, pred_speed_regional, std=self.drone_noise), 
                    min=0, max=14.0
                    )

        elif self.split == "test":
            # clean data for evaluation (before masking)
            logger.info("Using clean, noise-free, full-information labels for evaluation")
            pred_speed = torch.as_tensor(self._vdist_3min / self._vtime_3min, dtype=torch.float32)
            pred_speed_regional = torch.as_tensor(
                self.regional_speed_from_segment_numpy(self._vdist_3min, self._vtime_3min), 
                dtype=torch.float32
                )
        self.pred_speed = pred_speed
        self.pred_speed_regional = pred_speed_regional

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        """ 
        Get a sample with masking applied. This extends the parent __getitem__ method.
        
        In the files, we save the drone mask every 3 mins, since we assume the drones will move to monitor different locations every 3 minutes. However, the drone input is given every 5 seconds, so we need to extend the drone mask to the same time resolution as the drone input.
        
        The mask for loop detector is a 0-1 tensor of shape P, where P is the number of road segments. Since loop detectors are installed as fixed infrastructure, ld_mask remain unchanged over time.
        The mask for drone input of ONE sample is also a 0-1 tensor but it has shape (T, P), where P is the number of road segments, and T is the number of time steps. To avoid complex transition states, we assume drones to jump from one grid to another. 
        """
        data_dict = super().__getitem__(index)

        session_id = index // self.sample_per_session
        sample_id = index % self.sample_per_session

        drone_mask = self.drone_flight_mask[session_id, self.in_indexes_3min[sample_id], :]
        drone_mask = torch.repeat_interleave(drone_mask, self._r, dim=0)
        pred_mask = self.drone_flight_mask[session_id, self.out_indexes_3min[sample_id], :]

        data_dict['drone_mask'] = drone_mask
        data_dict['pred_mask'] = pred_mask
        data_dict['ld_mask'] = self.ld_mask
        
        return data_dict

    def load_or_compute_metadata(self):
        return super().load_or_compute_metadata()
    
    def collate_fn(self, list_of_seq: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_data = dict()
        for attr in self.io_seqs + self.add_seqs:
            batch_data[attr] = torch.cat(
                [seq[attr].unsqueeze(0) for seq in list_of_seq], dim=0
            ).contiguous()

        batch_data["metadata"] = self.metadata

        return batch_data