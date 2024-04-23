""" Generate simulation settings file and folders

    Example usage:
    python scripts/data/simbarca/gen_exp_settings.py --data_root datasets/simbarca --num_sim 100
"""
import os
import json
import shutil
import random
import itertools
import argparse
import numpy as np

def default_argument_parser():
    
    parser = argparse.ArgumentParser(description='Run simulations')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument('--init_seed', type=int, default=42, help='Initial seed for random number generator')
    parser.add_argument('--data_root', type=str, default='/home/weijiang/Projects/Netsanut/datasets/simbarca', help='Root directory for data')
    parser.add_argument('--num_thread', type=int, default=16, help='Number of threads to use when obtaining vehicle information')
    parser.add_argument('--num_sim', type=int, default=10, help='Number of simulations to run')
    parser.add_argument('--seed_low', type=int, default=10000, help='Lower bound for random seed')
    parser.add_argument('--seed_high', type=int, default=99999, help='Upper bound for random seed')
    
    parser.add_argument('--num_divs', default=5, type=int, help='Number of divisions for hyperparams')
    parser.add_argument('--global_scale_prob', type=float, default=0.9, help='Probability to apply global scaling')
    parser.add_argument('--global_scale_low', type=float, default=0.6, help='Lower bound for global scale')
    parser.add_argument('--global_scale_high', type=float, default=1.4, help='Upper bound for global scale')
    parser.add_argument('--mask_low', type=float, default=0.1, help='Lower bound for mask probability')
    parser.add_argument('--mask_high', type=float, default=0.3, help='Upper bound for mask probability')
    parser.add_argument('--noise_p_low', type=float, default=0.2, help='Lower bound for noise probability')
    parser.add_argument('--noise_p_high', type=float, default=0.5, help='Upper bound for noise probability')
    parser.add_argument('--noise_scale_low', type=float, default=0.1, help='Lower bound for noise scale')
    parser.add_argument('--noise_scale_high', type=float, default=0.3, help='Upper bound for noise scale')

    return parser
    
if __name__ == "__main__":
    
    parser = default_argument_parser()
    args = parser.parse_args()
    
    # a different seed for each simulation run
    rng = np.random.default_rng(seed=args.init_seed)
    sim_seeds = rng.integers(low=args.seed_low, high=args.seed_high, size=args.num_sim).tolist()
    
    # generate all combinations of hyperparams (and make the length longer than num_sim)
    mask_ps = np.linspace(args.mask_low, args.mask_high, args.num_divs).tolist()
    noise_ps = np.linspace(args.noise_p_low, args.noise_p_high, args.num_divs).tolist()
    noise_scales = np.linspace(args.noise_scale_low, args.noise_scale_high, args.num_divs).tolist()
    global_scales = np.linspace(args.global_scale_low, args.global_scale_high, args.num_divs).tolist()
    all_cfg_combs = list(itertools.product(mask_ps, noise_ps, noise_scales, global_scales))
    random.shuffle(all_cfg_combs)
    all_cfg_combs *= int(np.ceil(args.num_sim / len(all_cfg_combs)))
    
    # add the default one, no mask out, no noise, no global scaling
    sim_seeds = [42] + sim_seeds
    all_cfg_combs = [(0.0, 0.0, 0.0, 1.0)] + all_cfg_combs
    
    for idx, (seed, cfg) in enumerate(zip(sim_seeds, all_cfg_combs)):
        mask_p, noise_p, noise_scale, global_scale = cfg
        
        folder_name = "simulation_sessions/session_{:03d}".format(idx)
        folder_path = "{}/{}".format(args.data_root, folder_name)
        if os.path.exists(folder_path):
            if not args.overwrite:
                print("Folder {} already exists, skipping it".format(folder_path))
                continue
            else:
                print("Folder {} already exists, overwriting it".format(folder_path))
                shutil.rmtree(folder_path)

        os.makedirs(folder_path, exist_ok=False)
        os.mkdir("{}/{}".format(folder_path, "timeseries"))
        os.mkdir("{}/{}".format(folder_path, "trajectory"))

        # add variables to dictionary
        settings = {"random_seed": seed, 
                    "mask_p": mask_p, 
                    "noise_p": noise_p, 
                    "noise_scale": noise_scale, 
                    "global_scale": global_scale if rng.random() < args.global_scale_prob else 1.0,
                    "num_thread": args.num_thread}
        
        # save `settings` to json file
        settings_path = "{}/settings.json".format(folder_path)
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=4)
            

        
        