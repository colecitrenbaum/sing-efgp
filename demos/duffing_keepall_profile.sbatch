#!/bin/bash
# Compile-vs-execute split for the keep-all vs shimmed E-step, H100.
#
# WHY: post-Lever-A an EFGP fit is compile-dominated (T=10000: ~37s total, of
# which ~24.5s is one-time XLA compile -- see the fit-cost breakdown). The
# canonical Duffing walls (59s @ T=1k, 69s @ T=10k, 100s @ T=100k) are
# therefore ~55-60s of fixed compile plus a small T-dependent execution term.
# Keep-all builds a DIFFERENT E-step graph, so its compile cost differs too --
# a raw wall comparison conflates "different graph to compile" with "different
# math to execute". EFGP_PROFILE_ITERS=1 adds block_until_ready barriers so
# iter-0 (compile) separates from steady-state per-iter execution, which is the
# number that actually scales and the one a paper should quote.
#
# Submit:  BENCH_T=10000 sbatch demos/duffing_keepall_profile.sbatch
#SBATCH -p swl1,hns,owners
#SBATCH -C GPU_SKU:H100_SXM5
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=62G
#SBATCH --time=04:00:00
#SBATCH -o logs/duffkeepall_prof_%x_%j.out
set -uo pipefail
cd /scratch/users/ccitren/sing-efgp
export PIP_CACHE_DIR="$SCRATCH/.cache/pip" XDG_CACHE_HOME="$SCRATCH/.cache" \
       TMPDIR="$SCRATCH/.tmp" PYTHONNOUSERSITE=1
export EFGP_PROFILE_ITERS=1
ml purge; ml load python/3.12.1; ml load cuda/12.8.0
source /home/users/ccitren/venvs/sing-py312/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

T="${BENCH_T:-10000}"
OUT="${BENCH_OUT:-demos/_bench_duffing_keepall_prof}"

for m in efgp_keepall efgp; do
  echo "=================== PROFILED T=$T method=$m ==================="
  python -u demos/bench_duffing_scaling.py \
      --T "$T" --method "$m" --M 0 \
      --ls-init 0.7 --seed 0 --eps-grid 1e-3 --out-dir "$OUT"
done
