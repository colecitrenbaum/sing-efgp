#!/bin/bash
# Keep-all + gmix GATHER moments on the Duffing scaling bench, H100.
#
# gather_N=512 / stencil_r=64 is the coarsest resolution at which the fit does
# NOT diverge (64/8 and 256/32 both do -- see demos/diag_gather_grad_S.py and
# diag_gather_divergence.py). Production's 64/8 default is tuned for the shim,
# which needs only the moment value + m-gradient and absorbs ~5% error via
# rho-damping; keep-all feeds the gather's gradients straight into the
# precision J, so it needs sigma >> h_grid = 1/(gather_N*h_spec).
#
# Submit:  BENCH_T=1000 sbatch demos/duffing_keepall_gather_h100.sbatch
#SBATCH -p swl1,hns,owners
#SBATCH -C GPU_SKU:H100_SXM5
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=62G
#SBATCH --time=06:00:00
#SBATCH -o logs/duffkeepall_gather_%x_%j.out
set -uo pipefail
cd /scratch/users/ccitren/sing-efgp
export PIP_CACHE_DIR="$SCRATCH/.cache/pip" XDG_CACHE_HOME="$SCRATCH/.cache" \
       TMPDIR="$SCRATCH/.tmp" PYTHONNOUSERSITE=1
ml purge; ml load python/3.12.1; ml load cuda/12.8.0
source /home/users/ccitren/venvs/sing-py312/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

T="${BENCH_T:-1000}"
OUT="${BENCH_OUT:-demos/_bench_duffing_keepall_out}"
python -u demos/bench_duffing_scaling.py \
    --T "$T" --method efgp_keepall_gather --M 0 \
    --ls-init 0.7 --seed 0 --eps-grid 1e-3 \
    --gather-N 512 --gather-stencil-r 64 --out-dir "$OUT"
