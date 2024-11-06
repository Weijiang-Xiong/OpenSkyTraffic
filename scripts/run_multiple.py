import copy
import argparse
import subprocess
from itertools import product

RUN, VISUALIZE = 0, 1

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=int, default=0, choices=[0, 1], help="0 for run, 1 for visualize")
args = parser.parse_args()

mode = args.mode

# if mode == RUN:
#     common = "python tools/train.py --config-file config/NeTSFormer_uncertainty.py"
# elif mode == VISUALIZE:
#     common = "python tools/visualize.py --result-dir"
    
# datasets = ["metr-la", "pems-bay"]
# pp = [1,2,3]
# aa = [0.5, 1, 2, 3]

# command_list = []
# for dataset, p, a in product(datasets, pp, aa):
#     cmd = [common]
#     if mode == RUN:
#         cmd.append("data.dataset={} train.max_epoch=40".format(dataset))
#         cmd.append(
#             "train.milestone_cfg.model.loss.exponent={} train.milestone_cfg.model.loss.alpha={}".format(p, a)
#         )
#         cmd.append(
#             "train.output_dir=scratch/uncertainty_p{}a{}_{}".format(p, a, dataset).replace("0.5", "half").replace("-", "_")
#         )
#     elif mode == VISUALIZE:
#         cmd.append(
#         "scratch/uncertainty_p{}a{}_{}".format(p, a, dataset).replace("0.5", "half").replace("-", "_")
#         )
#     cmd_str = " ".join(cmd)
#     command_list.append(cmd_str)

train_script = "python scripts/train.py"
cfg_default = "--config-file config/HiMSNet.py"
cfg_rndobsv = "--config-file config/HiMSNetRND.py"

command_list = []

# command_list.append(
# f"{train_script} {cfg_default} train.output_dir=scratch/himsnet"
# )
# command_list.append(
# f"{train_script} {cfg_default} model.adjacency_hop=3 train.output_dir=scratch/himsnet_3hop"
# )
# command_list.append(
# f"{train_script} {cfg_default} model.adjacency_hop=5 train.output_dir=scratch/himsnet_5hop"
# )

"""
different training epochs
"""

# for training_epochs in range(60, 151, 30):
#     command_list.append(
#     f"{train_script} {cfg_default} model.adjacency_hop=3 train.max_epoch={training_epochs} train.output_dir=scratch/himsnet_ep{training_epochs}_3hop"
#     )

"""different model sizes (default is 64)
"""
# for model_size in [32, 128, 256]:
#     command_list.append(
#     f"{train_script} {cfg_default} model.adjacency_hop=3 model.d_model={model_size} train.output_dir=scratch/himsnet_d{model_size}_3hop"
#     )


""" 
Ablation on different data modalities when we have clean and noisy data.
We use 3-hop adjacency matrix as default as it is the best performing one
""" 
# command_list.append(
# f"{train_script} {cfg_default} train.output_dir=scratch/himsnet_no_emb_3hop model.adjacency_hop=3 model.simple_fillna=True"
# )
# command_list.append(
# f"{train_script} {cfg_default} model.use_drone=False model.adjacency_hop=3 train.output_dir=scratch/himsnet_no_drone_3hop"
# )
# command_list.append(
# f"{train_script} {cfg_default} model.use_ld=False model.adjacency_hop=3 train.output_dir=scratch/himsnet_no_ld_3hop"
# )


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


"""
different sensor penetration rates when we have random observations
"""
for percentage in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    command_list.append(
    f"{train_script} {cfg_rndobsv} model.adjacency_hop=3 train.output_dir=scratch/himsnet_rnd_noise_3hop_{int(100*percentage)}cvg dataset.train.use_clean_data=False dataset.train.ld_per={percentage} dataset.train.drone_per={percentage} dataset.test.ld_per={percentage} dataset.test.drone_per={percentage}"
    )


# evaluation commands
def find_output_dir(cmd_str):
    return cmd_str.split("output_dir=")[1].split(" ")[0]

eval_list = []
for cmd_str in command_list:
    eval_cmd = copy.deepcopy(cmd_str)
    output_dir = find_output_dir(eval_cmd)
    eval_cmd = eval_cmd.replace("train.py", "train.py --eval-only")
    eval_cmd = eval_cmd + " " + "train.checkpoint={}/model_final.pth".format(output_dir)
    eval_cmd = eval_cmd + " " + "evaluation.visualize=True"
    eval_list.append(eval_cmd)

if mode == VISUALIZE:
    command_list = eval_list

for cmd_str in command_list:
    try:
        print("Running command \n {}".format(cmd_str))
        completed_process = subprocess.run(cmd_str, shell=True)
    except Exception as e:
        print(e)
        continue
    