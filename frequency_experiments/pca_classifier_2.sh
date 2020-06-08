#!/bin/bash
#
#SBATCH --job-name=classifying
#SBATCH --array=0-999     ## N_TASKS_TOTAL: SET THIS TO THE NUMBER OF INDEPENDENT JOBS,TYPICLLAY 100,SEE BELOW N_TASKS_TOTAL
#SBATCH --output=/home/camp/warnert/bash_scripts/frequency_experiments/pca_class_%a.out
#SBATCH --error=/home/camp/warnert/bash_scripts/frequency_experiments/pca_class_%a.err
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=150G
#SBATCH --partition=cpu

python pca_classifier_2.py $SLURM_ARRAY_TASK_ID
