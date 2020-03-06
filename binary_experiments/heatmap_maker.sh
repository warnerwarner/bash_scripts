#!/bin/bash
#
#SBATCH --job-name=heatmaps
#SBATCH --array=1		## N_TASKS_TOTAL: SET THIS TO THE NUMBER OF INDEPENDENT JOBS,TYPICLLAY 100,SEE BELOW N_TASKS_TOTAL
#SBATCH --error=/home/camp/warnert/bash_scripts/binary_experiments/heatmap_out.txt
#SBATCH --error=/home/camp/warnert/bash_scripts/binary_experiments/heatmap_error.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --partition=cpu

python heatmap_maker.py
