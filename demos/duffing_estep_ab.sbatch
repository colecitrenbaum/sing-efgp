#!/bin/bash
# A/B of the two EFGP q(f) E-step paths on the Duffing scaling benchmark:
#   gmix     — pure-JAX Gaussian-mixture scatter + FFT (library GPU default)
#   analytic — cuFINUFFT type-1 + order-k Taylor envelope (library CPU default)
#
# Methodology note (why one job, not one job per cell): the canonical
# _bench_duffing_scaling_newcanon cells were submitted as 6 concurrent jobs that
# shared node sh04-05n05, so up to 4 of them held 4 different H100s at once.
# That is fine for a 3-5x EFGP-vs-SparseGP ratio but too noisy for an A/B where
# the effect may be tens of percent. Here every cell runs SEQUENTIALLY in ONE
# job on ONE H100, one fresh python process per cell (so no JIT-cache carryover
# between paths, matching the canonical per-cell process isolation).
#
# `wall` is timed around fit_efgp_sing_jax exactly as in the canonical cells,
# so it INCLUDES per-process JIT compile and is directly comparable to the
# published numbers. bench_analytic_gpu_parallel.py separates warm throughput
# from compile for the same two paths.
#
#SBATCH -p swl1
#SBATCH -C GPU_SKU:H100_SXM5
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --job-name=duff_estep_ab
#SBATCH --output=logs/estep_ab-%j.out
#SBATCH --error=logs/estep_ab-%j.err
set -uo pipefail; cd /scratch/users/ccitren/sing-efgp; mkdir -p logs
export PIP_CACHE_DIR="$SCRATCH/.cache/pip" XDG_CACHE_HOME="$SCRATCH/.cache" TMPDIR="$SCRATCH/.tmp" PYTHONNOUSERSITE=1
ml purge; ml load python/3.12.1; ml load cuda/12.8.0
source /home/users/ccitren/venvs/sing-py312/bin/activate

echo "=== node: $(hostname)  job: $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo

OUT=demos/_bench_duffing_estep_ab

# Warm-up cell into a throwaway dir: absorbs cold CUDA-context / NFS-import
# cost so the first MEASURED cell is not penalised for being first.
echo "### warmup (discarded)"
python -u demos/bench_duffing_scaling.py --T 400 --method efgp \
    --estep gmix --out-dir demos/_bench_duffing_estep_warmup
echo

# Paired cells: gmix immediately followed by analytic at each T, so the two
# paths see the same GPU/thermal/tenancy state.
for T in 1000 10000 100000; do
    for SPEC in "gmix 1" "analytic 1"; do
        set -- $SPEC
        ESTEP=$1; AORD=$2
        echo "### T=$T estep=$ESTEP order=$AORD"
        python -u demos/bench_duffing_scaling.py --T "$T" --method efgp \
            --estep "$ESTEP" --analytic-order "$AORD" --out-dir "$OUT"
        echo
    done
done

# analytic order 2 at the largest T only: the library docs warn order 1
# "regresses accuracy at large T", so this checks whether the accuracy gap (if
# any) is a Taylor-truncation artefact rather than intrinsic to the path.
echo "### T=100000 estep=analytic order=2"
python -u demos/bench_duffing_scaling.py --T 100000 --method efgp \
    --estep analytic --analytic-order 2 --out-dir "$OUT"

echo "=== all cells done ==="
