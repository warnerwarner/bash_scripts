#!/bin/bash
#
#SBATCH --job-name=setting_NN
#SBATCH --array=1,2,5
#SBATCH --output=/home/camp/warnert/bash_scripts/jULIE_recordings/NNsetting_%a.out
#SBATCH --error=/home/camp/warnert/bash_scripts/jULIE_recordings/NNsetting_%a.err
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=350G
#SBATCH --partition=cpu

python NN_recording_setting.py $SLURM_ARRAY_TASK_ID
