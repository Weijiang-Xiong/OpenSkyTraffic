#!/usr/bin/env zsh

python skymonitor/rl_drone.py --agent-type random --logdir ./scratch/drone_monitor_random;

python skymonitor/rl_drone.py --agent-type static --logdir ./scratch/drone_monitor_static;

repeats=${1:-5};

echo "Running PPO policies for ${repeats} repeats..."

for i in $(seq 1 ${repeats}); do
    base_seed=$((1000 + i * 10))

    python skymonitor/rl_drone.py --agent-type ppo --policy-type simple --train-seed ${base_seed} --logdir ./scratch/drone_monitor_ppo_simple_${i};

    python skymonitor/cleanrl_ppo_drone.py --train-seed ${base_seed} --logdir ./scratch/drone_monitor_cleanrl_ppo_simple_${i}

done
