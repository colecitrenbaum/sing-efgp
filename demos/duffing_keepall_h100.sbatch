#!/bin/bash
# Duffing scaling law on H100: keep-all autodiff E-step vs production shim vs
# SparseGP, all three run back-to-back on the SAME allocated GPU (fresh python
# process each) so wall times are directly comparable within a T.
#
# Submit:  BENCH_T=1000 sbatch demos/duffing_keepall_h100.sbatch
#SBATCH -p swl1,hns,owners
#SBATCH -C GPU_SKU:H100_SXM5
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=62G
#SBATCH --time=04:00:00
#SBATCH -o logs/duffkeepall_%x_%j.out
set -uo pipefail
cd /scratch/users/ccitren/sing-efgp
export PIP_CACHE_DIR="$SCRATCH/.cache/pip" XDG_CACHE_HOME="$SCRATCH/.cache" \
       TMPDIR="$SCRATCH/.tmp" PYTHONNOUSERSITE=1
ml purge; ml load python/3.12.1; ml load cuda/12.8.0
source /home/users/ccitren/venvs/sing-py312/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

T="${BENCH_T:-1000}"
OUT="${BENCH_OUT:-demos/_bench_duffing_keepall_out}"

# Re-run the two baselines alongside keep-all rather than quoting the older
# cells: same node, same branch, same batch -> nothing confounds the wall.
for spec in "efgp_keepall 0" "efgp 0" "sp 49"; do
  set -- $spec
  echo "=================== T=$T method=$1 M=$2 ==================="
  python -u demos/bench_duffing_scaling.py \
      --T "$T" --method "$1" --M "$2" \
      --ls-init 0.7 --seed 0 --eps-grid 1e-3 --out-dir "$OUT"
done
