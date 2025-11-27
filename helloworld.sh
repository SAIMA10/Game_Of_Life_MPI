#!/bin/bash 
#SBATCH --job-name=helloworld
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=02:00:00
#SBATCH --partition=devel
#SBATCH --account=teach0026
#SBATCH --output=helloworld.%j.out
#SBATCH --error=helloworld.%j.err

module purge 
module load miniconda/3
conda activate mpi

mpirun -np 4 python -m mpi4py helloworld.py
