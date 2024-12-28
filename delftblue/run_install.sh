#!/bin/sh
#
#SBATCH --job-name="cse-minor-install"
#SBATCH --partition=compute
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=1G

srun install.sh
