#!/bin/bash
#
#SBATCH --job-name=window_classifying
#SBATCH --output=/home/camp/warnert/bash_scripts/correlation_experiments/increasing_units.out
#SBATCH --error=/home/camp/warnert/bash_scripts/correlation_experiments/increasing_units.err
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --partition=cpu

python increasing_units.py
