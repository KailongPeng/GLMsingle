#!/bin/bash
#SBATCH -p psych_day,psych_gpu,psych_scavenge,psych_week
#SBATCH --job-name=example1
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=6:00:00
#SBATCH --output=logs/%J.out
#SBATCH --mem=100g
#SBATCH --requeue

set -e
module load FSL/5.0.10
. /nexsan/apps/hpc/Apps/FSL/5.0.10/etc/fslconf/fsl.sh
python -u /gpfs/milgram/project/turk-browne/projects/localize/analysis/GLMsingle/examples/example1.py

#cd /gpfs/milgram/project/turk-browne/projects/localize/analysis/GLMsingle/examples/
#sbatch example1.sh
