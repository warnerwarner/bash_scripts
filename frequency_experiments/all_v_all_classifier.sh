#!/bin/bash
#
#SBATCH --job-name=classifiying
#SBATCH --array=0-24	## N_TASKS_TOTAL: SET THIS TO THE NUMBER OF INDEPENDENT JOBS,TYPICLLAY 100,SEE BELOW N_TASKS_TOTAL
#SBATCH --output=/home/camp/warnert/bash_scripts/frequency_experiments/all_%a.out
#SBATCH --error=/home/camp/warnert/bash_scripts/frequency_experiments/all_%a.err
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=50G
#SBATCH --partition=cpu

python all_v_all_classifier.py $SLURM_ARRAY_TASK_ID
