#!/bin/bash
#
#SBATCH --job-name=PCA_classifing
#SBATCH --output=/home/camp/warnert/bash_scripts/correlation_experiments/increasing_PCA_comps.out
#SBATCH --error=/home/camp/warnert/bash_scripts/correlation_experiments/increasing_PCA_comps.err
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=150G
#SBATCH --partition=cpu

python PCA_classifier_ordered_single_comp.py $pca_components
