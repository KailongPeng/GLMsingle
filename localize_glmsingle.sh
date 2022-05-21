#!/bin/bash
#SBATCH -p psych_day,psych_gpu,psych_scavenge,psych_week
#SBATCH --job-name=glmsingle
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=6:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --mem=100g
#SBATCH --requeue

set -e
. /nexsan/apps/hpc/Apps/FSL/5.0.10/etc/fslconf/fsl.sh
python -u /gpfs/milgram/project/turk-browne/projects/localize/analysis/GLMsingle/localize_glmsingle.py ${SLURM_ARRAY_TASK_ID}

#subj=$1
#cd /gpfs/milgram/project/turk-browne/projects/localize/analysis/subjects/${subj}/
#read {subj}.txt,get the jobarrayID th line, get run
#runID=0
#while read line; do
#  echo $line;
#  runID=$((runID+1))
#  if runID==jobarrayID
#  then
#    running
#  fi
#done < ${subj}.txt

#running=func0${jobarrayID}
#echo $subj $running
#echo feat fsf.preprocess/${running}.fsf &
#feat fsf.preprocess/${running}.fsf &
#sleep 20
#bash scripts/wait-for-feat.sh preprocess/${running}.feat

