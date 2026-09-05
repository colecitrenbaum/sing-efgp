#!/bin/bash
#SBATCH -p swl1,hns,owners
#SBATCH -C GPU_SKU:H100_SXM5
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --job-name=verify_compile
#SBATCH --output=logs/verify_compile-%j.out
#SBATCH --error=logs/verify_compile-%j.err
set -uo pipefail; cd /scratch/users/ccitren/sing-efgp; mkdir -p logs
export PIP_CACHE_DIR="$SCRATCH/.cache/pip" XDG_CACHE_HOME="$SCRATCH/.cache" TMPDIR="$SCRATCH/.tmp" PYTHONNOUSERSITE=1
ml purge; ml load python/3.12.1; ml load cuda/12.8.0
source /home/users/ccitren/venvs/sing-py312/bin/activate
nvidia-smi --query-gpu=name --format=csv,noheader || true

echo "======================================================================"
echo "(1) CANONICAL EFGP REGRESSION SUBSET (CPU)"
echo "======================================================================"
JAX_PLATFORMS=cpu python -m pytest tests/test_efgp_em_multitrial.py \
  tests/test_efgp_jax_recovery.py tests/test_efgp_gmix_vs_mc.py \
  tests/test_efgp_analytic.py tests/test_efgp_sparsegp_drift_agreement.py \
  -q 2>&1 | tail -20

echo "======================================================================"
echo "(2) SMOOTHER COMPILE / JAXPR SIZE vs T  (CPU, compare to baseline table)"
echo "======================================================================"
JAX_PLATFORMS=cpu python -u demos/investigate_compile.py

echo "======================================================================"
echo "(3) FULL-FIT PER-ITER PROFILE  (H100; baseline was 75.4s, iter-0=62s)"
echo "======================================================================"
python -u demos/prof_iters_fit.py
