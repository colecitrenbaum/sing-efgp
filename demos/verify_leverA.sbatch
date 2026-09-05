#!/bin/bash
#SBATCH -p swl1,hns,owners
#SBATCH -C GPU_SKU:H100_SXM5
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --job-name=verify_leverA
#SBATCH --output=logs/verify_leverA-%j.out
#SBATCH --error=logs/verify_leverA-%j.err
set -uo pipefail; cd /scratch/users/ccitren/sing-efgp; mkdir -p logs
export PIP_CACHE_DIR="$SCRATCH/.cache/pip" XDG_CACHE_HOME="$SCRATCH/.cache" TMPDIR="$SCRATCH/.tmp" PYTHONNOUSERSITE=1
ml purge; ml load python/3.12.1; ml load cuda/12.8.0
source /home/users/ccitren/venvs/sing-py312/bin/activate
nvidia-smi --query-gpu=name --format=csv,noheader || true

echo "=================== CORRECTNESS: EFGP regression subset (CPU) ==================="
JAX_PLATFORMS=cpu python -m pytest tests/test_efgp_em_multitrial.py \
  tests/test_efgp_jax_recovery.py tests/test_efgp_gmix_vs_mc.py \
  tests/test_efgp_analytic.py tests/test_efgp_sparsegp_drift_agreement.py \
  -q 2>&1 | tail -20

echo "=================== WALL CLOCK: full-fit per-iter (H100) ==================="
echo "Baseline (pre-Lever-A): iter-0=62s, total=75.4s, ls=1.238"
python -u demos/prof_iters_fit.py
