#!/bin/bash
#
#SBATCH --job-name=array_printing
#SBATCH --array=1-10,11,15-20
#SBATCH --output=/home/camp/warnert/bash_scripts/misc_tests/array_%a.out
#SBATCH --error=/home/camp/warnert/bash_scripts/misc_tests/array_%a.err
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=1G
#SBATCH --partition=cpu

echo $SLURM_ARRAY_TASK_ID
