#!/usr/bin/env zsh

python skymonitor/rl_drone.py --agent-type random --logdir ./scratch/drone_monitor_random;

python skymonitor/rl_drone.py --agent-type static --logdir ./scratch/drone_monitor_static;

python skymonitor/rl_drone.py --agent-type ppo --policy-type simple --logdir ./scratch/drone_monitor_ppo_simple; 

python skymonitor/rl_drone.py --agent-type ppo --policy-type graph --logdir ./scratch/drone_monitor_ppo_graph; 

python skymonitor/rl_drone.py --agent-type ppo --policy-type grid --logdir ./scratch/drone_monitor_ppo_grid; 