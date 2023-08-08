import subprocess
from itertools import product

RUN, VISUALIZE = 0, 1

mode = VISUALIZE

if mode == RUN:
    common = "python tools/train.py --config-file config/NeTSFormer_uncertainty.py"
elif mode == VISUALIZE:
    common = "python tools/visualize.py --result-dir"
    
datasets = ["metr-la", "pems-bay"]
pp = [1,2,3]
aa = [0.5, 1, 2, 3]

command_list = []
for dataset, p, a in product(datasets, pp, aa):
    cmd = [common]
    if mode == RUN:
        cmd.append("data.dataset={} train.max_epoch=40".format(dataset))
        cmd.append(
            "train.milestone_cfg.model.loss.exponent={} train.milestone_cfg.model.loss.alpha={}".format(p, a)
        )
        cmd.append(
            "train.output_dir=scratch/uncertainty_p{}a{}_{}".format(p, a, dataset).replace("0.5", "half").replace("-", "_")
        )
    elif mode == VISUALIZE:
        cmd.append(
        "scratch/uncertainty_p{}a{}_{}".format(p, a, dataset).replace("0.5", "half").replace("-", "_")
        )
    cmd_str = " ".join(cmd)
    command_list.append(cmd_str)

for cmd_str in command_list:
    print("Running command \n {}".format(cmd_str))
    completed_process = subprocess.run(cmd_str, shell=True)