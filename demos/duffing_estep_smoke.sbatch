#!/bin/bash
# Smoke test for the --estep plumbing (gmix vs analytic) on an H100, tiny T.
# Purpose: catch signature/typo errors before the real A/B job runs for hours.
#SBATCH -p swl1,hns,owners
#SBATCH -C GPU_SKU:H100_SXM5
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=duff_estep_smoke
#SBATCH --output=logs/estep_smoke-%j.out
#SBATCH --error=logs/estep_smoke-%j.err
set -uo pipefail; cd /scratch/users/ccitren/sing-efgp; mkdir -p logs
export PIP_CACHE_DIR="$SCRATCH/.cache/pip" XDG_CACHE_HOME="$SCRATCH/.cache" TMPDIR="$SCRATCH/.tmp" PYTHONNOUSERSITE=1
ml purge; ml load python/3.12.1; ml load cuda/12.8.0
source /home/users/ccitren/venvs/sing-py312/bin/activate

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

OUT=demos/_bench_duffing_estep_smoke
for ESTEP in gmix analytic; do
    echo "### smoke estep=$ESTEP"
    python -u demos/bench_duffing_scaling.py --T 400 --method efgp \
        --estep "$ESTEP" --out-dir "$OUT"
done
echo "=== smoke done ==="
