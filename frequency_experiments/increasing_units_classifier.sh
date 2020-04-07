#!/bin/bash
#
#SBATCH --job-name=classifiying
#SBATCH --array=1	## N_TASKS_TOTAL: SET THIS TO THE NUMBER OF INDEPENDENT JOBS,TYPICLLAY 100,SEE BELOW N_TASKS_TOTAL
#SBATCH --output=/home/camp/warnert/bash_scripts/frequency_experiments/incr_class.out
#SBATCH --error=/home/camp/warnert/bash_scripts/frequency_experiments/incr_class.err
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=150G
#SBATCH --partition=cpu

python increasing_units_classifier.py
