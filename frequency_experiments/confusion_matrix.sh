#!/bin/bash
#
#SBATCH --job-name=confusing
#SBATCH --output=/home/camp/warnert/bash_scripts/frequency_experiments/confusion.out
#SBATCH --error=/home/camp/warnert/bash_scripts/frequency_experiments/confusion.err
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=50G
#SBATCH --partition=cpu

python confusion_matrix.py
