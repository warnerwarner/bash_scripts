#!/bin/bash
#
#SBATCH --job-name=PCA_classifing
#SBATCH --array=0-398    ## N_TASKS_TOTAL: SET THIS TO THE NUMBER OF INDEPENDENT JOBS,TYPICLLAY 100,SEE BELOW N_TASKS_TOTAL
#SBATCH --output=/home/camp/warnert/bash_scripts/correlation_experiments/increasing_PCA_comps_single_%a.out
#SBATCH --error=/home/camp/warnert/bash_scripts/correlation_experiments/increasing_PCA_comps_single_%a.err
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=150G
#SBATCH --partition=cpu

python PCA_classifier_ordered_single_comp.py $SLURM_ARRAY_TASK_ID
