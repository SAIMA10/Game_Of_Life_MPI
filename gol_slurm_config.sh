#!/bin/bash
#SBATCH --job-name=gol_mpi
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --time=03:00:00
#SBATCH --partition=devel
#SBATCH --account=teach0026
#SBATCH --output=gol_mpi.%j.out
#SBATCH --error=gol_mpi.%j.err

module purge
module load miniconda/3
conda activate mpi

mpirun -np $SLURM_NTASKS python -m mpi4py gol_2D.py \
  --nx 16384 --ny 16384 --steps 2000 --init random --seed 42 --print-every 100



