import copy
import argparse
import subprocess
from itertools import product

# evaluation commands
def find_output_dir(cmd_str):
    return cmd_str.split("output_dir=")[1].split(" ")[0]

def experiment_adjacency_hops(command_list):
    command_list.append(
    f"{train_script} {cfg_default} train.output_dir=scratch/himsnet"
    )
    command_list.append(
    f"{train_script} {cfg_default} model.adjacency_hop=3 train.output_dir=scratch/himsnet_3hop"
    )
    command_list.append(
    f"{train_script} {cfg_default} model.adjacency_hop=5 train.output_dir=scratch/himsnet_5hop"
    )
    
    return command_list

def experiment_epochs(command_list):
    """
    different training epochs
    """

    for training_epochs in range(60, 151, 30):
        command_list.append(
        f"{train_script} {cfg_default} model.adjacency_hop=3 train.max_epoch={training_epochs} train.output_dir=scratch/himsnet_ep{training_epochs}_3hop"
        )
    
    return command_list

def experiment_model_sizes(command_list):
    """different model sizes (default is 64)
    """
    for model_size in [32, 128, 256]:
        command_list.append(
        f"{train_script} {cfg_default} model.adjacency_hop=3 model.d_model={model_size} train.output_dir=scratch/himsnet_d{model_size}_3hop"
        )
    
    return command_list

def experiment_data_modality(command_list):
    """ 
    Ablation on different data modalities when we have clean and noisy data.
    We use 3-hop adjacency matrix as default as it is the best performing one
    """ 
    
    # # clean data
    command_list.append(
    f"{train_script} {cfg_default} train.output_dir=scratch/himsnet_no_emb_3hop model.adjacency_hop=3 model.simple_fillna=True"
    )
    command_list.append(
    f"{train_script} {cfg_default} model.use_ld=False model.adjacency_hop=3 train.output_dir=scratch/himsnet_no_ld_3hop"
    )
    command_list.append(
    f"{train_script} {cfg_default} model.use_drone=False model.adjacency_hop=3 train.output_dir=scratch/himsnet_no_drone_3hop"
    )
    command_list.append(
    f"{train_script} {cfg_default} model.use_global=False model.adjacency_hop=3 train.output_dir=scratch/himsnet_no_gnn_3hop"
    )
    
    # # partial and noisy data
    command_list.append(
    f"{train_script} {cfg_rndobsv} model.adjacency_hop=3 train.output_dir=scratch/himsnet_rnd_noise_fix_3hop dataset.train.use_clean_data=False"
    )
    command_list.append(
    f"{train_script} {cfg_rndobsv} model.adjacency_hop=3 model.use_drone=False train.output_dir=scratch/himsnet_rnd_no_drone_noise_fix_3hop dataset.train.use_clean_data=False"
    )
    command_list.append(
    f"{train_script} {cfg_rndobsv} model.adjacency_hop=3 model.use_ld=False train.output_dir=scratch/himsnet_rnd_no_ld_noise_fix_3hop dataset.train.use_clean_data=False"
    )
    command_list.append(
    f"{train_script} {cfg_rndobsv} model.adjacency_hop=3 model.use_global=False train.output_dir=scratch/himsnet_rnd_no_gnn_noise_fix_3hop dataset.train.use_clean_data=False"
    )
    command_list.append(
    f"{train_script} {cfg_rndobsv} model.adjacency_hop=3 model.simple_fillna=True train.output_dir=scratch/himsnet_rnd_no_emb_noise_fix_3hop dataset.train.use_clean_data=False"
    )

    return command_list


def experiment_penetration_rate(command_list):
    """
    different sensor penetration rates when we have random observations
    """
    for percentage in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        command_list.append(
        f"{train_script} {cfg_rndobsv} model.adjacency_hop=3 train.output_dir=scratch/himsnet_rnd_noise_3hop_{int(100*percentage)}cvg dataset.train.use_clean_data=False dataset.train.ld_cvg={percentage} dataset.train.drone_cvg={percentage} dataset.test.ld_cvg={percentage} dataset.test.drone_cvg={percentage}"
        )

    return command_list


def experiment_weight_factor(command_list):

    """
    different weight factors for the regional loss, a "." in the path may result in problems
    when parsing the checkpoint paths, we replace it with "p"
    """
    for weight in [0.2, 0.5, 1, 2, 5]:
        command_list.append(
        f"{train_script} {cfg_default} model.adjacency_hop=3 model.reg_loss_weight={weight} train.output_dir=scratch/himsnet_regional_loss_{str(weight).replace(".", "p")}_3hop"
        )

    return command_list

def experiment_emb_ablation_repeat(command_list):
    
    for r in range(10):
        command_list.append(
        f"{train_script} {cfg_default} model.adjacency_hop=3 train.output_dir=scratch/himsnet_3hop_r{r}"
        )
        # no emb
        command_list.append(
        f"{train_script} {cfg_default} model.adjacency_hop=3 model.simple_fillna=True train.output_dir=scratch/himsnet_no_emb_3hop_r{r}"
        )
        # noisy and partial data 
        command_list.append(
        f"{train_script} {cfg_rndobsv} model.adjacency_hop=3 train.output_dir=scratch/himsnet_rnd_noise_fix_3hop_r{r} dataset.train.use_clean_data=False"
        )
        # noisy and partial data no emb
        command_list.append(
        f"{train_script} {cfg_rndobsv} model.adjacency_hop=3 model.simple_fillna=True train.output_dir=scratch/himsnet_rnd_no_emb_noise_fix_3hop_r{r} dataset.train.use_clean_data=False"
        )
    
    return command_list

def experiment_lower_lr_for_ld_only(command_list):
    """
    Lower learning rate for the LD module only case with clean data
    
    But this does not help with the unstable training issue over the regional task
    """    
    command_list.append(
    f"{train_script} {cfg_default} model.adjacency_hop=3 optimizer.lr=0.00005 model.use_drone=False train.output_dir=scratch/himsnet_lr005_ld_3hop"
    )
    command_list.append(
    f"{train_script} {cfg_default} model.adjacency_hop=3 optimizer.lr=0.0001 model.use_drone=False train.output_dir=scratch/himsnet_lr01_ld_3hop"
    )
    command_list.append(
    f"{train_script} {cfg_default} model.adjacency_hop=3 optimizer.lr=0.0002 model.use_drone=False train.output_dir=scratch/himsnet_lr02_ld_3hop"
    )
    command_list.append(
    f"{train_script} {cfg_default} model.adjacency_hop=3 optimizer.lr=0.0005 model.use_drone=False train.output_dir=scratch/himsnet_lr05_ld_3hop"
    )
    command_list.append(
    f"{train_script} {cfg_default} model.adjacency_hop=3 optimizer.lr=0.0008 model.use_drone=False train.output_dir=scratch/himsnet_lr08_ld_3hop"
    )
    
    return command_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default=None, choices=["run", "vis"], help="training or visualization")
    args = parser.parse_args()
    
    mode = args.mode
    train_script = "python scripts/train.py"
    cfg_default = "--config-file config/HiMSNet.py"
    cfg_rndobsv = "--config-file config/HiMSNetRND.py"

    command_list = []

    # experiment_adjacency_hops(command_list)
    # experiment_epochs(command_list)
    # experiment_model_sizes(command_list)
    # experiment_data_modality(command_list)
    # experiment_penetration_rate(command_list)
    # experiment_weight_factor(command_list)
    # experiment_emb_ablation_repeat(command_list)
    experiment_lower_lr_for_ld_only(command_list)
    
    eval_list = []
    for cmd_str in command_list:
        eval_cmd = copy.deepcopy(cmd_str)
        output_dir = find_output_dir(eval_cmd)
        eval_cmd = eval_cmd.replace("train.py", "train.py --eval-only")
        eval_cmd = eval_cmd + " " + "train.checkpoint={}/model_final.pth".format(output_dir)
        eval_cmd = eval_cmd + " " + "evaluation.visualize=True"
        eval_list.append(eval_cmd)

    if mode == "vis":
        command_list = eval_list

    for cmd_str in command_list:
        try:
            print("Running command \n {}".format(cmd_str))
            completed_process = subprocess.run(cmd_str, shell=True)
        except Exception as e:
            print(e)
            continue
        