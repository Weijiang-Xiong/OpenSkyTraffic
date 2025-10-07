#!/bin/bash
#SBATCH --job-name RUN_EXP
#SBATCH --account=luts
#SBATCH --nodes 1
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --time 12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --output "out/slurm-%A_%a.log"
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

module load gcc/13.2.0 cuda/12.4.1
echo "${@:1}"
python -u "${@:1}"