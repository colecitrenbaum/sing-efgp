#!/bin/bash
#SBATCH -p swl1
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=gpdrift_iso_sweep
#SBATCH --output=logs/gpdrift_iso_sweep-%j.out
#SBATCH --error=logs/gpdrift_iso_sweep-%j.err

set -uo pipefail
cd /scratch/users/ccitren/sing-efgp
mkdir -p logs

export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export TMPDIR="$SCRATCH/.tmp"
export PYTHONNOUSERSITE=1

ml purge
ml load python/3.12.1
ml load cuda/12.8.0
source /home/users/ccitren/venvs/sing-py312/bin/activate

python -u demos/bench_gpdrift_inducing_sweep_iso_x64.py
