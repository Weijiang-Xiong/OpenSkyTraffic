""" 
This script is used to run multiple experiments with different configurations.
It generates a list of commands to run the experiments or to visualize the results. 
The script can be run in different modes: "run", "vis", or "both". 
The generated commands can be run as sbatch jobs or directly in the terminal. 
The commands are grouped into different experiments based on the configuration parameters.

To run the experiments locally on a computer, use the following command. The commands will be run one by one, with the training going first and testing afterwards. 
    `python scripts/run_multiple.py --mode both`

However, since a slurm cluster run the jobs in parallel, we can't use `both` mode as a testing must be run after the training. 
    `python scripts/run_multiple.py --mode run`
"""
import glob
import copy
import argparse
import subprocess

TRAIN_SCRIPT = "python scripts/train.py"

# evaluation commands
def find_output_dir(cmd_str):
    return cmd_str.split("output_dir=")[1].split(" ")[0]

def run_as_config(train_script, cfg_file, command_list):
    command_list.append(
    f"{train_script} --config-file {cfg_file}"
    )
    return command_list


def experiment_simbarca_gmm_model_data_modality(train_script, cfg_file, command_list, exp):
    experiments = []

    experiments.append(
    f"{train_script} --config-file {cfg_file} train.output_dir=scratch/{exp}_gmmpred"
    )
    experiments.append(
    f"{train_script} --config-file {cfg_file} model.use_drone=False train.output_dir=scratch/{exp}_gmmpred_ld_only"
    )
    experiments.append(
    f"{train_script} --config-file {cfg_file} model.use_ld=False train.output_dir=scratch/{exp}_gmmpred_drone_only"
    )

    # in the random observation case, we try to add a scenario with very high noise over the labels 
    # this is to simulate the case where we have no drones at all, and the high-quality labels are not possible
    # we have noisy ones instead 
    if "rnd" in exp: 
        experiments.append(
        f"{train_script} --config-file {cfg_file} model.use_drone=False dataset.train.drone_noise=0.3 train.output_dir=scratch/{exp}_gmmpred_ld_only_very_noisy"
        )

    command_list.extend(experiments)

    return command_list

def experiment_adapted_gmm_models(command_list, train_script):
    """ Run default settings as written in the config file, without any changes
    """

    # run the default settings
    experiments = []
    
    for folder in ['lgc', 'stid', 'staeformer', 'mtgnn', 'gwnet']:
        config_files = glob.glob(f"config/{folder}/*.py")
        for config_file in config_files:
            experiments.append(f"{train_script} --config-file {config_file}")
    
    command_list.extend(experiments)

    return command_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", type=str, choices=["run", "vis", "both"], help="training or visualization")
    parser.add_argument("--use-sbatch", action="store_true", help="whether to run as sbatch jobs")
    args = parser.parse_args()
    
    mode = args.mode
    command_list = []

    # experiment_adapted_gmm_models(command_list, TRAIN_SCRIPT)
    experiment_simbarca_gmm_model_data_modality(TRAIN_SCRIPT, "config/himsnet/HiMSNet_GMM_RND.py", command_list, "simbarca_rnd")
    experiment_simbarca_gmm_model_data_modality(TRAIN_SCRIPT, "config/himsnet/HiMSNet_GMM_Full.py", command_list, "simbarca_full")
    for cfg in ["HiMSNet", "HiMSNet_RND"]:
        run_as_config(TRAIN_SCRIPT, f"config/himsnet/{cfg}.py", command_list)


    if mode in ["vis", "both"]:
        eval_list = []
        for cmd_str in command_list:
            eval_cmd = copy.deepcopy(cmd_str)
            eval_cmd = eval_cmd.replace("train.py", "train.py --eval-only")
            try:
                output_dir = find_output_dir(eval_cmd)
                eval_cmd = eval_cmd + " " + "train.checkpoint={}/model_final.pth".format(output_dir)
            except Exception:
                print("Output directory not specified, relying on the default save dir for evaluation")
            eval_list.append(eval_cmd)

    if mode == "vis":
        command_list = eval_list
    elif mode == "both":
        command_list = command_list + eval_list

    if args.use_sbatch:
        if mode == "both":
            raise ValueError("Cannot run the training and visualization jobs in parallel, please use run mode")
        command_list = [cmd_str.replace("python", "sbatch python_wrapper.sh") for cmd_str in command_list]

    print("Will run the following commands:\n")
    for cmd_str in command_list:
        print(cmd_str)

    if input("\ncontinue? (y/n)") not in ["y", "Y", "yes", "Yes", "YES"]:
        exit()

    for cmd_str in command_list:
        try:
            print("Running command \n {}".format(cmd_str))
            completed_process = subprocess.run(cmd_str, shell=True)
        except Exception as e:
            print(e)
            continue
