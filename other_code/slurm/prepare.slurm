#!/bin/bash
#SBATCH --account=vsharan_1861
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/prepare_%j.out

module purge
module load gcc/13.3.0
module load python/3.11.9

cd /scratch1/ssinha05/mlproject

python3 -m venv venv
source venv/bin/activate
pip install torch datasets transformers scikit-learn numpy tqdm

python3 scripts/prepare_data.py
