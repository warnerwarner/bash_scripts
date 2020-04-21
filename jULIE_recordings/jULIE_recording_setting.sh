#!/bin/bash
#
#SBATCH --job-name=setting_jULIE
#SBATCH --output=/home/camp/warnert/bash_scripts/jULIE_recordings/setting.out
#SBATCH --error=/home/camp/warnert/bash_scripts/jULIE_recordings/setting.err
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=250G
#SBATCH --partition=cpu

python jULIE_recordings_setting.py
