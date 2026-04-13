#!/bin/bash
#SBATCH --account=vsharan_1861
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/smoke_%j.out

module purge
module load gcc/13.3.0
module load python/3.11.9

cd /scratch1/ssinha05/mlproject
source venv/bin/activate

python3 scripts/smoke_test.py