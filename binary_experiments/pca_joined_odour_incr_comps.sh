#!/bin/bash
#
#SBATCH --job-name=classifying
#SBATCH --array=0-100	## N_TASKS_TOTAL: SET THIS TO THE NUMBER OF INDEPENDENT JOBS,TYPICLLAY 100,SEE BELOW N_TASKS_TOTAL
#SBATCH --output=/home/camp/warnert/bash_scripts/binary_experiments/pca_comps%a.out
#SBATCH --error=/home/camp/warnert/bash_scripts/binary_experiments/pca_comps%a.err
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --partition=cpu

python pca_joined_odour_incr_comps.py $SLURM_ARRAY_TASK_ID
