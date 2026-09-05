#!/bin/bash
#SBATCH -p swl1,hns,owners
#SBATCH -C GPU_SKU:H100_SXM5
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --job-name=cold_fit
#SBATCH --output=logs/cold_fit-%j.out
#SBATCH --error=logs/cold_fit-%j.err
set -uo pipefail; cd /scratch/users/ccitren/sing-efgp; mkdir -p logs
export PIP_CACHE_DIR="$SCRATCH/.cache/pip" XDG_CACHE_HOME="$SCRATCH/.cache" TMPDIR="$SCRATCH/.tmp" PYTHONNOUSERSITE=1
ml purge; ml load python/3.12.1; ml load cuda/12.8.0
source /home/users/ccitren/venvs/sing-py312/bin/activate
nvidia-smi --query-gpu=name --format=csv,noheader || true
python -u demos/cold_fit.py
