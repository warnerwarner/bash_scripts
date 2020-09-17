#!/bin/bash
#
#SBATCH --job-name=classifying
#SBATCH --array=1-145	## N_TASKS_TOTAL: SET THIS TO THE NUMBER OF INDEPENDENT JOBS,TYPICLLAY 100,SEE BELOW N_TASKS_TOTAL
#SBATCH --output=/home/camp/warnert/bash_scripts/binary_experiments/basic_classifier_%a.out
#SBATCH --error=/home/camp/warnert/bash_scripts/binary_experiments/basic_classifier_%a.err
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --partition=cpu

python basic_classifiers.py $SLURM_ARRAY_TASK_ID
