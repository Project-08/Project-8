#!/bin/sh
#
#SBATCH --job-name="cse-minor-project8"
#SBATCH --partition=gpu-a100-small
#SBATCH --time=00:02:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G

module load 2024r1
module unload cuda/11.6
module load cuda/12.5
module load miniconda3

srun project8.sh
