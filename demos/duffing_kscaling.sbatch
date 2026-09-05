#!/bin/bash
# Parametrized single-cell GPU job for the Duffing K-sweep (number of trials).
# Cell params via env (set by submit_duffing_kscaling.sh): K, T, METHOD, M, SEED,
# EPS, OUTDIR. Per-cell --mem/--time/-J supplied on the sbatch command line.
#SBATCH -p swl1
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --job-name=duff_kscale
#SBATCH --output=logs/duff_kscale-%j.out
#SBATCH --error=logs/duff_kscale-%j.err
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

: "${K:?set K}"
: "${METHOD:?set METHOD}"
T="${T:-1000}"
M="${M:-0}"
SEED="${SEED:-0}"
LSINIT="${LSINIT:-0.7}"
EPS="${EPS:-0.001}"
KMINLS="${KMINLS:-}"          # optional: force EFGP grid to resolve down to this l
NOLEARNK="${NOLEARNK:-}"      # optional: 1 -> freeze kernel hypers (M-step off)
VARINIT="${VARINIT:-}"        # optional: fixed/init kernel variance
RESTOREQF="${RESTOREQF:-}"    # optional: none|hutch|hutch_hetS (restore E[V])
OUTDIR="${OUTDIR:-demos/_bench_duffing_kscaling_out}"

KMIN_ARG=""
[ -n "$KMINLS" ] && KMIN_ARG="--k-min-ls $KMINLS"
[ -n "$NOLEARNK" ] && KMIN_ARG="$KMIN_ARG --no-learn-kernel"
[ -n "$VARINIT" ] && KMIN_ARG="$KMIN_ARG --var-init $VARINIT"
[ -n "$RESTOREQF" ] && KMIN_ARG="$KMIN_ARG --restore-qf-var $RESTOREQF"

echo "cell: K=$K T=$T METHOD=$METHOD M=$M SEED=$SEED EPS=$EPS KMINLS=$KMINLS OUTDIR=$OUTDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u demos/bench_duffing_kscaling.py \
    --K "$K" --T "$T" --method "$METHOD" --M "$M" \
    --ls-init "$LSINIT" --seed "$SEED" --eps-grid "$EPS" $KMIN_ARG --out-dir "$OUTDIR"
