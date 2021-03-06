#!/bin/bash
#
#SBATCH --job-name=window_classifying
#SBATCH --array=0-399   ## N_TASKS_TOTAL: SET THIS TO THE NUMBER OF INDEPENDENT JOBS,TYPICLLAY 100,SEE BELOW N_TASKS_TOTAL
#SBATCH --output=/home/camp/warnert/bash_scripts/correlation_experiments/increasing_window_%a.out
#SBATCH --error=/home/camp/warnert/bash_scripts/correlation_experiments/increasing_window_%a.err
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --partition=cpu

python window_classifiers.py $SLURM_ARRAY_TASK_ID
