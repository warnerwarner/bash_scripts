#!/bin/bash
#
#SBATCH --job-name=PCA_classifing
#SBATCH --array=0-999    ## N_TASKS_TOTAL: SET THIS TO THE NUMBER OF INDEPENDENT JOBS,TYPICLLAY 100,SEE BELOW N_TASKS_TOTAL
#SBATCH --output=/home/camp/warnert/bash_scripts/correlation_experiments/split_PCA_comps_%a.out
#SBATCH --error=/home/camp/warnert/bash_scripts/correlation_experiments/split_PCA_comps_%a.err
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=150G
#SBATCH --partition=cpu

python PCA_seperate_increasing_comps.py $SLURM_ARRAY_TASK_ID
