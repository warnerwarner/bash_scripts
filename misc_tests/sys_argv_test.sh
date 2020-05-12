#!/bin/bash
#
#SBATCH --job-name=PCA_classifing
#SBATCH --output=/home/camp/warnert/bash_scripts/misc_tests/sys_arv.out
#SBATCH --error=/home/camp/warnert/bash_scripts/misc_tests/sys_arv.err
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=1G
#SBATCH --partition=cpu

python sys_arv_test.py 1 2 3 4 5 6 7 8 9 10 
