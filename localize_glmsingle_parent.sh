#!/bin/bash
#SBATCH -p psych_day,psych_gpu,psych_scavenge,psych_week
#SBATCH --job-name=glmsingle
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=6:00:00
#SBATCH --output=logs/%J.out
#SBATCH --mem=15g
#SBATCH --requeue

cd /gpfs/milgram/project/turk-browne/projects/localize/analysis/GLMsingle/
set -e
. /nexsan/apps/hpc/Apps/FSL/5.0.10/etc/fslconf/fsl.sh
module load miniconda
conda activate glmsingle
python -u /gpfs/milgram/project/turk-browne/projects/localize/analysis/GLMsingle/localize_glmsingle_parent.py ${SLURM_ARRAY_TASK_ID}
