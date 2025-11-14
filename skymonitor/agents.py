from enum import Enum
from typing import Dict, List, Tuple

import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn

from skymonitor.policy_net import DronePolicy

class DroneAction(Enum):
    """ A drone can stay, move up/down/left/right by at most 2 steps, or move diagonally by 1 step.
        E.g., a drone can fly at 20 m/s, and a grid cell is 220 m, in a time step of 3 mins, 
        a drone can can spend ~20s to move and the rest of the time to hover and collect data.
    """
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class RandomAgent:

    def __init__(self, action_space: spaces.Space):
        self.action_space = action_space

    def select_action(self, obs: Dict) -> List[DroneAction]:
        sampled = self.action_space.sample()
        return [DroneAction(int(x)) for x in sampled]


class CentralizedMonitoringAgent:

    def __init__(self, num_drones: int, grid: Dict = None):
        self.policy_net: DronePolicy = None
        self.num_drones = int(num_drones)
        self.positions: List[Tuple[int, int]] = None
        self.action_space: gym.Space = None
        # the grid_id of all road segments, shape (num_locations,)
        self.grid_id = grid.get("grid_id", None)
        # the grid (x, y) coordinates of all road segments, shape (num_locations, 2)
        self.grid_xy = grid.get("grid_xy", None)
        # Dict[Tuple[int, int], int], the mapping from (x, y) to grid_id
        # the keys are non-empty grid (x,y) coordinates, the values are the ids of non-empty grids
        self.grid_xy_to_id = grid.get("grid_xy_to_id", None)
        self.id_of_non_empty_grids = list(self.grid_xy_to_id.values())
        self.xy_of_non_empty_grids = list(self.grid_xy_to_id.keys())

    def bind_action_space(self, action_space: spaces.Space) -> None:
        """ The action space of the agent should match that of the environment. """
        self.action_space = action_space

    def select_action(self, obs: Dict) -> List[DroneAction]:
        """ Choose next drone actions given observation for all drones.
            Args:
                obs: Dict, the observation from the environment, from function `TrafficMonitorEnv.build_observation`
                info: Dict, the info from `TrafficMonitorEnv.step`

            return: List[DroneAction], the actions for all drones
        """
        # shape (in_steps, num_locations, feature_dim), the observed traffic for the recent time step (3 mins)
        # drone data has high frequency (5s), so we have in_steps = 36 for 3 mins
        # in feature_dim, we have (flow, density, time-in-day), time-in-day is normalized to [0,1], e.g., 1/3 means 8am
        current_traffic = obs['observed_traffic']
        # shape (num_locations,), a True means the location is covered by drones
        coverage_mask = obs['coverage_mask']
        # predicted traffic for next 30 minutes, with 3 min time step 
        # shape (batch_size, out_steps, num_locations, feature_dim)
        predicted_traffic = obs['batch_pred'].squeeze()

        predicted_traffic_at_grid_x9y7 = predicted_traffic[:, self.grid_id==self.grid_xy_to_id[(9,7)], :]
        flow_at_x9y7 = predicted_traffic_at_grid_x9y7[..., 0] # shape (out_steps, road_segments_in_grid_x9y7)
        density_at_x9y7 = predicted_traffic_at_grid_x9y7[..., 1]  # shape (out_steps, road_segments_in_grid_x9y7)

        
        predicted_traffic_at_grid_id = predicted_traffic[:, self.grid_id==self.id_of_non_empty_grids[77], :]
        flow_at_grid_id_42 = predicted_traffic_at_grid_id[..., 0]  # shape (out_steps, road_segments_in_grid_id_42)
        density_at_grid_id_42 = predicted_traffic_at_grid_id[..., 1]  # shape (out_steps, road_segments_in_grid_id_42)

        sampled = self.action_space.sample()
        return [DroneAction(int(x)) for x in sampled]

    def update(self, experiences: List[Tuple]) -> None:
        """Update policy from collected experience (RL training)."""
        return None
    
