#!/bin/bash
#
#SBATCH --job-name=setting_jULIE
#SBATCH --array=0-5
#SBATCH --output=/home/camp/warnert/bash_scripts/jULIE_recordings/setting_%a.out
#SBATCH --error=/home/camp/warnert/bash_scripts/jULIE_recordings/setting_%a.err
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=250G
#SBATCH --partition=cpu

python jULIE_recording_setting.py $SLURM_ARRAY_TASK_ID
