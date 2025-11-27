#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=example_code
#SBATCH --time=02:00:00
#SBATCH --partition=devel
#SBATCH --account=teach0026
#SBATCH --output=example_code.%j.out
#SBATCH --error=example_code.%j.err

module purge
module load miniconda/3

python example_code.py 5
