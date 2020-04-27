#!/bin/bash
#
#SBATCH --job-name=setting_jULIE
#SBATCH --array=0-31
#SBATCH --output=/home/camp/warnert/bash_scripts/jULIE_recordings/unique_spikes_0211_%a.out
#SBATCH --error=/home/camp/warnert/bash_scripts/jULIE_recordings/unique_spikes_0211_%a.err
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --partition=cpu

python unique_spike_finder_190211.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT
