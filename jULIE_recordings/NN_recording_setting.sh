#!/bin/bash
#
#SBATCH --job-name=setting_NN
#SBATCH --array=3,4
#SBATCH --output=/home/camp/warnert/bash_scripts/jULIE_recordings/setting_%a.out
#SBATCH --error=/home/camp/warnert/bash_scripts/jULIE_recordings/setting_%a.err
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=500G
#SBATCH --partition=hmem

python NN_recording_setting.py $SLURM_ARRAY_TASK_ID
