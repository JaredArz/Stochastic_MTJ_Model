#!/bin/bash
#SBATCH --job-name=leap_tests
#SBATCH --cpus-per-task=1
#SBATCH --array=0-24%25
#SBATCH --output=runs/output/run_%A-%a.txt
#SBATCH --error=runs/errors/run_%A-%a.txt

sleep 10
srun -n 1 python -u STT_RNG_Leap_SingleProc.py --ID $SLURM_ARRAY_TASK_ID