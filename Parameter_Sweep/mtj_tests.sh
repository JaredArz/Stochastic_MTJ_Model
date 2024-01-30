#!/bin/bash
#SBATCH --job-name=mtj_tests
#SBATCH --cpus-per-task=2
##SBATCH --array=0-19999%200
#SBATCH --array=0-6560%200
#SBATCH --output=runs/output/run_%A-%a.txt
#SBATCH --error=runs/errors/run_%A-%a.txt

sleep 10
srun -n 1 python mtj_param_sweep.py --ID $SLURM_ARRAY_TASK_ID