#!/bin/bash
#
#SBATCH --job-name=classifiyin
#SBATCH --array=0-399		## N_TASKS_TOTAL: SET THIS TO THE NUMBER OF INDEPENDENT JOBS,TYPICLLAY 100,SEE BELOW N_TASKS_TOTAL
#SBATCH --output=/home/camp/warnert/bash_scripts/frequency_experiments/window_class_diff_%a.out
#SBATCH --error=/home/camp/warnert/bash_scripts/frequency_experiments/window_class_diff_%a.err
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=150G
#SBATCH --partition=cpu

python window_classifier_difference.py $SLURM_ARRAY_TASK_ID
