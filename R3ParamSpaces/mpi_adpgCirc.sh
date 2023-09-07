#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --partition=short
#SBATCH --job-name=adpgCirc
#SBATCH --ntasks=100
#SBATCH --time=04:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=F_r_e@hotmail.es
#SBATCH --error=logs/err-%j.log
#SBATCH --output=logs/out-%j.log
##------------------------ End job description ------------------------

srun python mpi_adpgCirc.py


