#!/bin/bash
#
#SBATCH --job-name=classifying
#SBATCH --array=27,28,29,30,31,32,38,39,42,43,44,46,47,49,55,59	## N_TASKS_TOTAL: SET THIS TO THE NUMBER OF INDEPENDENT JOBS,TYPICLLAY 100,SEE BELOW N_TASKS_TOTAL
#SBATCH --output=/home/camp/warnert/bash_scripts/binary_experiments/pca_classifier_%a.out
#SBATCH --error=/home/camp/warnert/bash_scripts/binary_experiments/pca_classifier_%a.err
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --partition=cpu

python pca_classifiers.py $SLURM_ARRAY_TASK_ID
