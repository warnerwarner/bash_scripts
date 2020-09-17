#!/bin/bash
#
#SBATCH --job-name=sniffing
#SBATCH --array=1		## N_TASKS_TOTAL: SET THIS TO THE NUMBER OF INDEPENDENT JOBS,TYPICLLAY 100,SEE BELOW N_TASKS_TOTAL
#SBATCH --output=/home/camp/warnert/bash_scripts/binary_experiments/sniffing.out
#SBATCH --error=/home/camp/warnert/bash_scripts/binary_experiments/sniffing.err
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --partition=cpu

python sniff_basis_maker.py