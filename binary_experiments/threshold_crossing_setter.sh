#!/bin/bash
#
#SBATCH --job-name=thresholding
#SBATCH --array=1		## N_TASKS_TOTAL: SET THIS TO THE NUMBER OF INDEPENDENT JOBS,TYPICLLAY 100,SEE BELOW N_TASKS_TOTAL
#SBATCH --output=/home/camp/warnert/bash_scripts/binary_experiments/thresh_out_%j.out
#SBATCH --error=/home/camp/warnert/bash_scripts/binary_experiments/thresh_error_%j.err
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=350G
#SBATCH --partition=hmem

python heatmap_maker.py
