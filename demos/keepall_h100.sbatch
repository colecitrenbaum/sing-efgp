#!/bin/bash
# Keep-all autodiff E-step (gmix_full_batched) vs production (gmix_batched)
# on the Duffing benchmark, H100. T is passed as $1 via --export or sbatch args.
#SBATCH -p swl1,hns,owners
#SBATCH -C GPU_SKU:H100_SXM5
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=62G
#SBATCH --time=04:00:00
#SBATCH -o logs/keepall_h100_%x_%j.out
set -uo pipefail
cd /scratch/users/ccitren/sing-efgp
export PIP_CACHE_DIR="$SCRATCH/.cache/pip" XDG_CACHE_HOME="$SCRATCH/.cache" \
       TMPDIR="$SCRATCH/.tmp" PYTHONNOUSERSITE=1
ml purge; ml load python/3.12.1; ml load cuda/12.8.0
source /home/users/ccitren/venvs/sing-py312/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -u demos/bench_keepall_autodiff.py "${BENCH_T:-10000}" "${BENCH_NEM:-15}"
