#!/bin/sh
#
#SBATCH --job-name="cse-minor-install"
#SBATCH --partition=compute
#SBATCH --time=00:25:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=1G

module load 2024r1
module unload cuda/11.6
module load cuda/12.5
module load miniconda3
module load cmake

srun install.sh
