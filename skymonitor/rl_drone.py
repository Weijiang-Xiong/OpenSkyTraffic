""" We consider flying a fleet of drones to collect traffic data (flow, density, speed) over a large urban area.
    The goal is to smartly plan the drone flights so that the collected data can allow a traffic predictor to achieve the best performance.
    We plan to develop an RL-based solution to this problem, which involves:
        1. A dataset `SimBarcaExplore` that provides traffic data of an urban area, where the space is divided to grids.
        2. A traffic predictor that takes in past observations and predict the future traffic states (flow, density, speed)
        3. A reward calculator based on the evaluation of traffic predictions (how good is the predictor doing)
        4. A set of monitoring agents (drones) to query data from the dataset
        5. An environment to orchestrate the four components above
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from skytraffic.utils.event_logger import setup_logger
from skymonitor.simbarca_explore import SimBarcaExplore
from skymonitor.agents import CentralizedMonitoringAgent, DroneAction, DronePolicy
from skymonitor.monitor_env import TrafficMonitorEnv
from skymonitor.traffic_predictor import TrafficPredictor

logger = setup_logger(name="default", log_file="./scratch/rl_drone.log", level=logging.INFO)

def train_monitoring_agent_with_ppo(
    total_timesteps: int = 100_000,
    num_envs: int = 1,
    action_space: int = 10,
    learning_rate: float = 3e-4,
    log_dir: str = "scratch/ppo_monitoring_agent",
    seed: int = 0,
    dataset_kwargs: Optional[Dict] = None,
    predictor_device: str = "cpu",
) -> PPO:
    """Train PPO on the monitoring environment with the custom policy."""
    dataset_defaults = dict(
        split="train",
        input_window=3,
        pred_window=30,
        step_size=3,
        num_unpadded_samples=20,
        allow_shorter_input=False,
        pad_input=False,
        norm_tid=False,
    )
    dataset_config = dataset_defaults if dataset_kwargs is None else {**dataset_defaults, **dataset_kwargs}

    set_random_seed(seed)
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    def _make_env(rank: int):
        def _init():
            env_dataset = SimBarcaExplore(**dataset_config)
            env_predictor = TrafficPredictor(device=predictor_device)
            env = TrafficMonitorEnv(dataset=env_dataset, predictor=env_predictor, num_drones=num_drones)
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env

        return _init

    vec_env = DummyVecEnv([_make_env(rank) for rank in range(num_envs)])

    model = PPO(
        policy=DronePolicy,
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=max(32, 1024 // max(1, num_envs)),
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=str(log_path),
        verbose=1,
        seed=seed,
    )

    model.learn(total_timesteps=total_timesteps)
    checkpoint_path = log_path / "ppo_monitoring_agent"
    model.save(checkpoint_path)
    vec_env.close()
    return model

def collect_pred_and_gt(env: TrafficMonitorEnv, agent: CentralizedMonitoringAgent) -> Tuple[torch.Tensor, torch.Tensor]:

    all_pred, all_gt = [], []
    for active_session in range(env.dataset.num_sessions):
        print("=== Running agents on session {} ===".format(active_session))
        observation, info = env.reset(options={"active_session": active_session})
        done = False
        truncated = False
        episode_reward = 0.0
        step_count = 0
        episode_pred, episode_gt = [torch.as_tensor(observation['batch_pred']).squeeze()], [torch.as_tensor(info['batch_gt']).squeeze()]
        while not (done or truncated):
            actions = agent.select_action(observation)
            observation, reward, done, truncated, info = env.step(actions)
            step_count += 1
            episode_reward += reward
            episode_pred.append(torch.as_tensor(observation['batch_pred'].squeeze())) # T, P, C
            episode_gt.append(torch.as_tensor(info['batch_gt'].squeeze())) # T, P, C
        
        all_pred.append(torch.stack(episode_pred, dim=0))  # stack on new batch dimension
        all_gt.append(torch.stack(episode_gt, dim=0))
    
    all_pred = torch.cat(all_pred, dim=0) # concatenate on existing batch dimension
    all_gt = torch.cat(all_gt, dim=0)

    return all_pred, all_gt

if __name__ == "__main__":
    num_drones = 10
    num_vec_env = 3
    dataset = SimBarcaExplore(
        split="train",
        input_window=3,
        pred_window=30,
        step_size=3,
        num_unpadded_samples=20,
        allow_shorter_input=False,
        pad_input=False,
        norm_tid=False,
        vectorized=num_vec_env is not None,
    )
    predictor = TrafficPredictor(device='cuda') # looks like I don't really need this wrapper class ...
    
    action_space = spaces.MultiDiscrete([len(DroneAction)] * num_drones)
    
    env = TrafficMonitorEnv(
        dataset=dataset,
        predictor=predictor,
        action_space=action_space,
        num_vec_env=num_vec_env,
    )

    from stable_baselines3.common.env_checker import check_env
    check_env(env, warn=True)

    grid_info = {"grid_xy": dataset.grid_xy, "grid_id": dataset.grid_id, "grid_xy_to_id": dataset.grid_xy_to_id}
    agent = CentralizedMonitoringAgent(action_space=action_space, grid=grid_info)

    agent.vectorize_action_space(num_vec_env=num_vec_env)
    observation, info = env.reset()
    env.step(agent.select_action(observation))  # test step

    # Example PPO usage:
    # train_monitoring_agent_with_ppo(total_timesteps=50_000, num_envs=2, num_drones=num_drones, predictor_device='cuda')

    observation, info = env.reset()
    logger.info("Initial observation keys:{}".format(observation.keys()))
    logger.info("Initial coverage ratio: {}".format(observation["coverage_mask"].mean()))

    # test the agent, not vectorized
    dataset = SimBarcaExplore(
        split="test",
        input_window=3,
        pred_window=30,
        step_size=3,
        num_unpadded_samples=20,
        allow_shorter_input=False,
        pad_input=False,
        norm_tid=False,
        vectorized=False,
    )
    env = TrafficMonitorEnv(
        dataset=dataset,
        predictor=predictor,
        action_space=action_space,
        num_vec_env=None,
    )
    agent = CentralizedMonitoringAgent(action_space=action_space, grid=grid_info)
    with torch.no_grad():
        all_pred, all_gt = collect_pred_and_gt(env, agent)

    from skymonitor.simbarca_explore_evaluation import SimBarcaExploreEvaluator
    evaluator = SimBarcaExploreEvaluator(
        save_dir="scratch/rl_drone_evaluation",
        visualize=True,
        ignore_value=0.0,
    )
    eval_results = evaluator.calculate_error_metrics(pred=all_pred, label=all_gt, data_channels=dataset.data_channels['target'])
    logger.info("Drone Monitoring Evaluation Results: {}".format(eval_results))

    env.close()