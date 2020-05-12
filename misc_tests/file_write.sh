#!/bin/bash
#
#SBATCH --job-name=file_writing
#SBATCH --output=/home/camp/warnert/bash_scripts/misc_tests/file_write.out
#SBATCH --error=/home/camp/warnert/bash_scripts/misc_tests/file_write.err
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=1G
#SBATCH --partition=cpu

python file_write.py
