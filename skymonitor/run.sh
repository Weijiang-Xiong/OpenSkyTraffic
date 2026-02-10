#!/usr/bin/env zsh

python skymonitor/rl_drone.py --agent-type random --logdir ./scratch/drone_monitor_random;

python skymonitor/rl_drone.py --agent-type static --logdir ./scratch/drone_monitor_static;

repeats=${1:-5}
echo "Running PPO policies for ${repeats} repeats..."

for i in $(seq 1 ${repeats}); do
    python skymonitor/rl_drone.py --agent-type ppo --policy-type simple --logdir ./scratch/drone_monitor_ppo_simple_${i};

    python skymonitor/rl_drone.py --agent-type ppo --policy-type graph --logdir ./scratch/drone_monitor_ppo_graph_${i}; 

    python skymonitor/rl_drone.py --agent-type ppo --policy-type grid --logdir ./scratch/drone_monitor_ppo_grid_${i}; 

done