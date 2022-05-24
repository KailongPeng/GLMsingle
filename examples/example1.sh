#!/bin/bash
#SBATCH -p psych_day,psych_gpu,psych_scavenge,psych_week
#SBATCH --job-name=glmsingle
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=6:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --mem=100g
#SBATCH --requeue

set -e
module load FSL/5.0.10
. /nexsan/apps/hpc/Apps/FSL/5.0.10/etc/fslconf/fsl.sh
python -u /gpfs/milgram/project/turk-browne/projects/localize/analysis/GLMsingle/examples/example1.py
