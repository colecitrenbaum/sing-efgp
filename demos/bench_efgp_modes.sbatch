#!/bin/bash
#SBATCH -p swl1
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=efgp_modes
#SBATCH --output=logs/efgp_modes-%j.out
#SBATCH --error=logs/efgp_modes-%j.err

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

python -u demos/bench_efgp_modes_x64.py
