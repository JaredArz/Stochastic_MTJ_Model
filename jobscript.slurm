#!/bin/bash

#SBATCH -J f90_parallel          # Job name
#SBATCH -o f90_parallel.o%j       # Name of stdout output file
#SBATCH -e f90_parallel.e%j       # Name of stderr error file
#SBATCH -p normal           # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 00:01:00        # Run time (hh:mm:ss)
#SBATCH -A OTH21017     # Project allocation
#SBATCH --mail-user=jared.arzate@utexas.edu
#SBATCH --mail-type=all    # Send email at begin and end of job

# Other commands must follow all #SBATCH directives...

pwd
date

# Launch code
srun -c 128 python mtj_config_testing.py
