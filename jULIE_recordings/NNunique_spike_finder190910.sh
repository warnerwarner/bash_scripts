#!/bin/bash
#
#SBATCH --job-name=unique_spikes
#SBATCH --array=0-31
#SBATCH --output=/home/camp/warnert/bash_scripts/jULIE_recordings/NNunique_spikes_190910%a.out
#SBATCH --error=/home/camp/warnert/bash_scripts/jULIE_recordings/NNunique_spikes_190910%a.err
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --partition=cpu

python NNunique_spike_finder190910.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT
