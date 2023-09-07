#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --partition=short
#SBATCH --job-name=braak
#SBATCH --ntasks=100
#SBATCH --time=06:30:00
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_50
#SBATCH --mail-user=F_r_e@hotmail.es
#SBATCH --error=logs/err-%j.log
#SBATCH --output=logs/out-%j.log
##------------------------ End job description ------------------------

srun python mpi_braak.py


