from enum import Enum
from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from skymonitor.monitor_env import MapStructure
    from stable_baselines3.common.policies import BasePolicy
    
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

class BaseAgent:

    def __init__(self):
        self.action_space: spaces.Space

    def select_action(self, obs: Dict) -> np.ndarray | List[np.ndarray]:
        """ Choose next drone actions given observation for all drones.
            Args:
                obs: Dict, the observation from the environment, from function `TrafficMonitorEnv.build_observation`
                info: Dict, the info from `TrafficMonitorEnv.step`

            return: List[DroneAction], the actions for all drones
        """
        raise NotImplementedError

class StaticAgent(BaseAgent):

    def __init__(self, num_drones: int = 10):
        self.action_space = spaces.MultiDiscrete(nvec=[len(DroneAction)] * num_drones)
        self.static_action = [DroneAction.STAY.value] * num_drones

    def select_action(self, obs: Dict) -> np.ndarray | List:
        return self.static_action


class RandomAgent(BaseAgent):

    def __init__(self, num_drones: int = 10):
        self.action_space = spaces.MultiDiscrete(nvec=[len(DroneAction)] * num_drones)

    def select_action(self, obs: Dict) -> np.ndarray | List:
        return self.action_space.sample()


class PolicyAgent(BaseAgent):
    """ Just for the interface of `select_action`, possibly some logging can be added here in the future.
    """

    def __init__(self, policy = None):
        self.policy = policy

    def select_action(self, obs: Dict) -> np.ndarray | List:
        """ Let a policy network select actions based on observation.
        """
        # use the policy net to select actions
        actions, _states = self.policy.predict(observation=obs)

        return actions