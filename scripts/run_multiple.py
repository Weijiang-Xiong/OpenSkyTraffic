import subprocess
from itertools import product

# RUN, VISUALIZE = 0, 1

# mode = VISUALIZE

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
command_list.append(
f"{train_script} {cfg_default} train.output_dir=scratch/himsnet"
)
command_list.append(
f"{train_script} {cfg_default} model.use_global=False train.output_dir=scratch/himsnet_no_gnn"
)
command_list.append(
f"{train_script} {cfg_default} model.use_drone=False train.output_dir=scratch/himsnet_no_drone"
)
command_list.append(
f"{train_script} {cfg_default} model.use_ld=False train.output_dir=scratch/himsnet_no_ld"
)
command_list.append(
f"{train_script} {cfg_default} model.adjacency_hop=3 train.output_dir=scratch/himsnet_3hop"
)
command_list.append(
f"{train_script} {cfg_default} model.adjacency_hop=5 train.output_dir=scratch/himsnet_5hop"
)
command_list.append(
f"{train_script} {cfg_default} train.max_epoch=80 train.output_dir=scratch/himsnet_ep80"
)
command_list.append(
f"{train_script} {cfg_default} model.d_model=128 train.output_dir=scratch/himsnet_d128"
)
command_list.append(
f"{train_script} {cfg_default} model.d_model=256 train.output_dir=scratch/himsnet_d256"
)

command_list.append(
f"{train_script} {cfg_rndobsv} train.output_dir=scratch/himsnet_rnd"
)
command_list.append(
f"{train_script} {cfg_rndobsv} model.use_drone=False train.output_dir=scratch/himsnet_rnd_no_drone"
)
command_list.append(
f"{train_script} {cfg_rndobsv} model.use_ld=False train.output_dir=scratch/himsnet_rnd_no_ld"
)

for cmd_str in command_list:
    print("Running command \n {}".format(cmd_str))
    completed_process = subprocess.run(cmd_str, shell=True)