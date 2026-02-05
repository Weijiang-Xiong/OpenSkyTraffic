from enum import Enum
from collections import deque
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

ActionToDirection = {
    DroneAction.STAY: (0, 0),
    DroneAction.UP: (0, 1),
    DroneAction.DOWN: (0, -1),
    DroneAction.LEFT: (-1, 0),
    DroneAction.RIGHT: (1, 0),
}

DirectionToAction = {v: k for k, v in ActionToDirection.items()}

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

class SweepingAgent(BaseAgent):

    """ A sweeping-scan agent that follows a pre-computed sweeping-scan plan.
        sweeping_pos: np.ndarray, the grid positions (x, y) that the agent should sweep, shape (num_grid, 2)
    """

    def __init__(self, num_drones: int = 10, sweeping_pos: np.ndarray = None):
        super().__init__()
        self.num_drones = num_drones
        self.action_space = spaces.MultiDiscrete(nvec=[len(DroneAction)] * num_drones)
        self.trajectory_plan: List[List[Tuple[int, int]]] = self.build_sweep_scan_plan(sweeping_pos, num_drones)
        self.trajectory_progress: List[int] = [None] * num_drones  # track progress along each drone's trajectory

    def select_action(self, obs):
        """ Select next actions based on the sweeping-scan plan.
        """
        actions = [] 
        for agent_id in range(self.num_drones):
            
            # find where the agents are currently located on the planned trajectory
            agent_pos = (int(obs['positions_x'][agent_id]), int(obs['positions_y'][agent_id]))
            agent_progress = self.trajectory_progress[agent_id]
            if agent_progress is None:
                agent_progress = self.trajectory_plan[agent_id].index(agent_pos)
                self.trajectory_progress[agent_id] = agent_progress

            if agent_progress < len(self.trajectory_plan[agent_id]) - 1:
                next_pos = self.trajectory_plan[agent_id][agent_progress + 1]
                direction = (next_pos[0] - agent_pos[0], next_pos[1] - agent_pos[1])
                action = DirectionToAction[direction]
                actions.append(action)
                self.trajectory_progress[agent_id] = agent_progress + 1

            elif agent_progress == len(self.trajectory_plan[agent_id]) - 1:
                # revert the trajectory and go backward to the starting point
                self.trajectory_plan[agent_id] = list(reversed(self.trajectory_plan[agent_id]))
                next_pos = self.trajectory_plan[agent_id][1]
                direction = (next_pos[0] - agent_pos[0], next_pos[1] - agent_pos[1])
                action = DirectionToAction[direction]
                actions.append(action)
                self.trajectory_progress[agent_id] = 1

        return actions

    def clear_state(self):
        """ Clear any internal state if necessary.
        """
        self.trajectory_progress = [None] * self.num_drones
    
    @staticmethod
    def build_sweep_scan_plan(grid_xy: np.ndarray, num_drones: int) -> List[List[Tuple[int, int]]]:
        """Generate a sweeping-scan plan for multiple agents.

        The plan sweeps row-by-row in a serpentine pattern across available cells.
        Each agent gets a contiguous chunk of the sweep order, and moves one grid
        step at a time via shortest paths on available cells.
        """
        positions = {(int(x), int(y)) for x, y in np.asarray(grid_xy)}
        if not positions:
            return [[] for _ in range(num_drones)]

        xs = [pos[0] for pos in positions]
        ys = [pos[1] for pos in positions]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        sweep_targets: List[Tuple[int, int]] = []
        for y in range(min_y, max_y + 1):
            row = [(x, y) for x in range(min_x, max_x + 1) if (x, y) in positions]
            if not row:
                continue
            if (y - min_y) % 2 == 0:
                sweep_targets.extend(row)
            else:
                sweep_targets.extend(reversed(row))

        def _split_evenly(items: List[Tuple[int, int]], parts: int) -> List[List[Tuple[int, int]]]:
            if parts <= 0:
                return []
            base = len(items) // parts
            extra = len(items) % parts
            chunks = []
            start = 0
            for idx in range(parts):
                size = base + (1 if idx < extra else 0)
                chunks.append(items[start : start + size])
                start += size
            return chunks

        def _shortest_path(start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
            if start == goal:
                return [start]
            queue = deque([start])
            came_from = {start: None}
            while queue and goal not in came_from:
                cx, cy = queue.popleft()
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    nxt = (nx, ny)
                    if nxt in positions and nxt not in came_from:
                        came_from[nxt] = (cx, cy)
                        queue.append(nxt)

            if goal not in came_from:
                raise ValueError(f"No path between {start} and {goal} on the available grid.")

            path = [goal]
            cur = goal
            while came_from[cur] is not None:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path

        chunks = _split_evenly(sweep_targets, num_drones)
        trajectories: List[List[Tuple[int, int]]] = []
        for chunk in chunks:
            if not chunk:
                trajectories.append([])
                continue
            path = [chunk[0]]
            for target in chunk[1:]:
                segment = _shortest_path(path[-1], target)
                path.extend(segment[1:])
            trajectories.append(path)

        if len(trajectories) < num_drones:
            filler = trajectories[-1][-1] if trajectories and trajectories[-1] else sweep_targets[-1]
            for _ in range(num_drones - len(trajectories)):
                trajectories.append([filler])

        return trajectories

class PolicyAgent(BaseAgent):
    """ Just for the interface of `select_action`, possibly some logging can be added here in the future.
    """

    def __init__(self, policy: 'BasePolicy' = None):
        self.policy = policy

    def select_action(self, obs: Dict) -> np.ndarray | List:
        """ Let a policy network select actions based on observation.
        """
        # use the policy net to select actions
        actions, _states = self.policy.predict(observation=obs)

        return actions