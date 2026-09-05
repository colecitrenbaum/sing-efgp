#!/bin/bash
# Parametrized single-cell GPU job for the GP-drift scaling law.
# Cell params come in as env vars (set by submit_gpdrift_scaling.sh via
# `sbatch --export`): T, METHOD, M, SEED, OUTDIR, LSINIT.
# Per-cell --mem / --time / -J are supplied on the sbatch command line and
# override the modest defaults below.
#SBATCH -p swl1
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --job-name=gpd_scaling
#SBATCH --output=logs/gpd_scaling-%j.out
#SBATCH --error=logs/gpd_scaling-%j.err
# Pin to ONE GPU model so wall times are comparable across all cells. swl1 has
# both A100_SXM4 (node sh03-17n12) and H100_SXM5 (node sh04-05n05); without this
# constraint jobs land on either -> confounded timings. submit script overrides
# via -C. Both SKUs are 80GB.
#SBATCH -C GPU_SKU:H100_SXM5

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

: "${T:?set T}"
: "${METHOD:?set METHOD}"
M="${M:-0}"
SEED="${SEED:-0}"
LSINIT="${LSINIT:-0.7}"
EPS="${EPS:-0.001}"
RUNNER="${RUNNER:-demos/bench_duffing_scaling.py}"
OUTDIR="${OUTDIR:-demos/_bench_duffing_scaling_out}"

echo "cell: RUNNER=$RUNNER T=$T METHOD=$METHOD M=$M SEED=$SEED LSINIT=$LSINIT EPS=$EPS OUTDIR=$OUTDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u "$RUNNER" \
    --T "$T" --method "$METHOD" --M "$M" \
    --ls-init "$LSINIT" --seed "$SEED" --eps-grid "$EPS" --out-dir "$OUTDIR"
