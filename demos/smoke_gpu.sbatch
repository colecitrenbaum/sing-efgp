#!/bin/bash
#SBATCH -p dev
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
# NOTE: dev rejects `sbatch` (interactive-only) AND caps per-user memory (QOS).
# Run this on dev via: srun -p dev -G 1 -c 4 --mem 16G -t 00:20:00 bash demos/smoke_gpu.sbatch
#SBATCH --job-name=sing_smoke_gpu
#SBATCH --output=logs/smoke_gpu-%j.out
#SBATCH --error=logs/smoke_gpu-%j.err

set -uo pipefail
cd /scratch/users/ccitren/sing-efgp
mkdir -p logs

# Keep caches + user-site isolation (see setup_sing_venv.sh).
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export TMPDIR="$SCRATCH/.tmp"
export PYTHONNOUSERSITE=1

ml purge
ml load python/3.12.1
ml load cuda/12.8.0
source /home/users/ccitren/venvs/sing-py312/bin/activate

python -u demos/smoke_gpu.py
