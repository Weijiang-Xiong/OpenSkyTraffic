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

# evaluation commands
def find_output_dir(cmd_str):
    return cmd_str.split("output_dir=")[1].split(" ")[0]

def experiment_adjacency_hops(command_list):
    
    for hop in [1, 3, 5, 7, 9]:
        command_list.append(
        f"{train_script} --config-file {cfg_default} model.adjacency_hop={hop} train.output_dir=scratch/himsnet_{hop}hop"
        )
    
    return command_list

def experiment_epochs(command_list):
    """
    different training epochs
    """

    for training_epochs in range(60, 151, 30):
        command_list.append(
        f"{train_script} {cfg_default} model.adjacency_hop=5 train.max_epoch={training_epochs} train.output_dir=scratch/himsnet_ep{training_epochs}_5hop"
        )
    
    return command_list

def experiment_model_sizes(command_list):
    """different model sizes (default is 64)
    """
    for model_size in [32, 128, 256]:
        command_list.append(
        f"{train_script} {cfg_default} model.adjacency_hop=5 model.d_model={model_size} train.output_dir=scratch/himsnet_d{model_size}_5hop"
        )
    
    return command_list

def experiment_data_modality(command_list):
    """ 
    Ablation on different data modalities when we have clean and noisy data.
    We use 3-hop adjacency matrix as default as it is the best performing one
    """ 
    
    # # clean data
    command_list.append(
    f"{train_script} {cfg_default} train.output_dir=scratch/himsnet_no_emb_5hop model.adjacency_hop=5 model.simple_fillna=True"
    )
    command_list.append(
    f"{train_script} {cfg_default} model.use_ld=False model.adjacency_hop=5 train.output_dir=scratch/himsnet_no_ld_5hop"
    )
    command_list.append(
    f"{train_script} {cfg_default} model.use_drone=False model.adjacency_hop=5 train.output_dir=scratch/himsnet_no_drone_5hop"
    )
    command_list.append(
    f"{train_script} {cfg_default} model.use_global=False model.adjacency_hop=5 train.output_dir=scratch/himsnet_no_gnn_5hop"
    )
    
    # # partial and noisy data
    command_list.append(
    f"{train_script} {cfg_rndobsv} model.adjacency_hop=5 train.output_dir=scratch/himsnet_rnd_noise_fix_5hop dataset.train.use_clean_data=False"
    )
    command_list.append(
    f"{train_script} {cfg_rndobsv} model.adjacency_hop=5 model.use_drone=False train.output_dir=scratch/himsnet_rnd_no_drone_noise_fix_5hop dataset.train.use_clean_data=False"
    )
    command_list.append(
    f"{train_script} {cfg_rndobsv} model.adjacency_hop=5 model.use_ld=False train.output_dir=scratch/himsnet_rnd_no_ld_noise_fix_5hop dataset.train.use_clean_data=False"
    )
    command_list.append(
    f"{train_script} {cfg_rndobsv} model.adjacency_hop=5 model.use_global=False train.output_dir=scratch/himsnet_rnd_no_gnn_noise_fix_5hop dataset.train.use_clean_data=False"
    )
    command_list.append(
    f"{train_script} {cfg_rndobsv} model.adjacency_hop=5 model.simple_fillna=True train.output_dir=scratch/himsnet_rnd_no_emb_noise_fix_5hop dataset.train.use_clean_data=False"
    )

    return command_list

def experiment_norm_and_attn_agg(command_list):
    """
    Ablation on different normalization and attention aggregation methods
    """
    command_list.append(
    f"{train_script} {cfg_default} model.adjacency_hop=5 model.normalize_input=False model.attn_agg=False train.output_dir=scratch/himsnet_5hop_no_norm_avgagg"
    )
    command_list.append(
    f"{train_script} {cfg_default} model.adjacency_hop=5 model.normalize_input=False model.attn_agg=True train.output_dir=scratch/himsnet_5hop_no_norm_attnagg"
    )
    command_list.append(
    f"{train_script} {cfg_default} model.adjacency_hop=5 model.normalize_input=True model.attn_agg=False train.output_dir=scratch/himsnet_5hop_norm_avgagg"
    )
    
    return command_list

def experiment_tf_glb(command_list):
    """ Use transformer encoder for global message exchange
    """
    command_list.append(
    f"{train_script} {cfg_default} model.adjacency_hop=5 model.tf_glb=True train.output_dir=scratch/himsnet_5hop_tf_glb"
    )
    command_list.append(
    f"{train_script} {cfg_default} model.adjacency_hop=5 model.tf_glb=False train.output_dir=scratch/himsnet_5hop_gnn_glb"
    )
    return command_list

def experiment_penetration_rate(command_list):
    """
    different sensor penetration rates when we have random observations
    """
    for percentage in [0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        command_list.append(
        f"{train_script} {cfg_rndobsv} model.adjacency_hop=5 train.output_dir=scratch/himsnet_rnd_noise_5hop_{int(100*percentage)}cvg dataset.train.use_clean_data=False dataset.train.ld_cvg={percentage} dataset.train.drone_cvg={percentage} dataset.test.ld_cvg={percentage} dataset.test.drone_cvg={percentage}"
        )
        
    return command_list


def experiment_weight_factor(command_list):

    """
    different weight factors for the regional loss, a "." in the path may result in problems
    when parsing the checkpoint paths, we replace it with "p"
    """
    for weight in [0.2, 0.5, 1, 2, 5]:
        command_list.append(
        f'{train_script} {cfg_default} model.adjacency_hop=5 model.reg_loss_weight={weight} train.output_dir=scratch/himsnet_regional_loss_{str(weight).replace(".", "p")}_5hop'
        )

    return command_list

def experiment_emb_ablation(train_script, cfg_file, command_list, exp):
    # this part shows EMB has no significant impact on the performance
    for r in range(3):
        command_list.append(
        f"{train_script} {cfg_file} model.adjacency_hop=5 train.output_dir=scratch/{exp}_r{r}"
        )
        # no emb
        command_list.append(
        f"{train_script} {cfg_file} model.adjacency_hop=5 model.simple_fillna=True train.output_dir=scratch/{exp}_no_emb_r{r}"
        )
    
    return command_list


def experiment_gmm_model(train_script, cfg_file, command_list, exp):
    command_list.append(
    f"{train_script} --config-file {cfg_file} model.adjacency_hop=5 model.map_estimation=False train.output_dir=scratch/gmmpred_bayes_avg_{exp}"
    )
    command_list.append(
    f"{train_script} --config-file {cfg_file} model.adjacency_hop=5 model.map_estimation=False model.use_drone=False train.output_dir=scratch/gmmpred_bayes_avg_no_drone_{exp}"
    )
    command_list.append(
    f"{train_script} --config-file {cfg_file} model.adjacency_hop=5 model.map_estimation=False model.use_ld=False train.output_dir=scratch/gmmpred_bayes_avg_no_ld_{exp}"
    )
    command_list.append(
    f"{train_script} --config-file {cfg_file} model.adjacency_hop=5 model.map_estimation=True train.output_dir=scratch/gmmpred_map_est_{exp}"
    )
    command_list.append(
    f"{train_script} --config-file {cfg_file} model.adjacency_hop=5 model.map_estimation=True model.use_drone=False train.output_dir=scratch/gmmpred_map_est_no_drone_{exp}"
    )
    command_list.append(
    f"{train_script} --config-file {cfg_file} model.adjacency_hop=5 model.map_estimation=True model.use_ld=False train.output_dir=scratch/gmmpred_map_est_no_ld_{exp}"
    )

def experiment_gmm_models(command_list, train_script, cfg_file, exp, overrides:dict=None):
    experiments = []
    experiments.append(
    f"{train_script} --config-file {cfg_file}_GMMSingle.py train.output_dir=scratch/{exp}_lgc_single"
    )
    experiments.append(
    f"{train_script} --config-file {cfg_file}_GMM.py train.output_dir=scratch/{exp}_lgc_gmm"
    )
    experiments.append(
    f"{train_script} --config-file {cfg_file}.py train.output_dir=scratch/{exp}_lgc"
    )

    if overrides is not None:
        override_str = " ".join([f"{k}={v}" for k, v in overrides.items()])
        experiments = [cmd_str + " " + override_str for cmd_str in experiments]

    command_list.extend(experiments)
    return command_list

def experiment_adapted_models(command_list, train_script):
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
    train_script = "python scripts/train.py"
    cfg_default = "config/HiMSNet.py"
    cfg_rndobsv = "config/HiMSNetRND.py"

    command_list = []

    # experiment_adjacency_hops(command_list)
    # experiment_epochs(command_list)
    # experiment_model_sizes(command_list)
    # experiment_data_modality(command_list)
    # experiment_norm_and_attn_agg(command_list)
    # experiment_tf_glb(command_list)
    # experiment_penetration_rate(command_list)
    # experiment_weight_factor(command_list)
    # experiment_emb_ablation(train_script, cfg_default, command_list, experiment_name="himsnet_5hop")
    # experiment_emb_ablation(train_script, cfg_rndobsv, command_list, experiment_name="himsnet_rnd_noise_fix_5hop")
    # experiment_gmm_model(train_script, cfg_rndobsv, command_list, experiment_name="rndobs")
    # experiment_gmm_model(train_script, cfg_default, command_list, experiment_name="fullinfo")
    
    # experiment_gmm_models(command_list, train_script, "LSTMGCNConv", "metr")
    # experiment_gmm_models(command_list, train_script, "config/lgc/SimbarcaSpd_LSTMGCNConv", "simbarcaspd") 
    # experiment_gmm_models(command_list, train_script, "PEMSBAY_LSTMGCNConv", "pemsbay")
    experiment_adapted_models(command_list, train_script)


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
            eval_cmd = eval_cmd + " " + "evaluator.visualize=True"
            eval_list.append(eval_cmd)

    if mode == "vis":
        command_list = eval_list
    elif mode == "both":
        command_list = command_list + eval_list

    if args.use_sbatch:
        if mode == "both":
            raise ValueError("Cannot run the training and visualization jobs in parallel, please use run mode")
        command_list = [cmd_str.replace("python", "sbatch python_wrapper.sh") for cmd_str in command_list]

    print("Will run the following commands:")
    for cmd_str in command_list:
        print(cmd_str)

    for cmd_str in command_list:
        try:
            print("Running command \n {}".format(cmd_str))
            completed_process = subprocess.run(cmd_str, shell=True)
        except Exception as e:
            print(e)
            continue
        