#!/bin/bash
# Run-to-run reproducibility probe at T=100000, same seed, same H100.
#
# Why: the A/B job's gmix cell at T=1e5 reproduced the published newcanon WALL
# (102.4s vs 100.4s) but NOT the published drift NRMSE (0.0824 vs 0.0953) or
# lengthscale (1.368 vs 1.576) -- at identical code (newcanon ran after HEAD
# 559d6ae), identical seed, identical GPU model. T=1e3 and T=1e4 matched to 4
# digits, so the suspicion is nondeterministic float accumulation order in the
# gmix spreader's atomic scatter, amplified over 50 EM iters and worst at the
# largest source count. Repeats into separate out-dirs (same seed) settle it,
# and the analytic repeats show whether the cuFINUFFT path shares the problem.
#
#SBATCH -p swl1
#SBATCH -C GPU_SKU:H100_SXM5
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=duff_estep_rep
#SBATCH --output=logs/estep_rep-%j.out
#SBATCH --error=logs/estep_rep-%j.err
set -uo pipefail; cd /scratch/users/ccitren/sing-efgp; mkdir -p logs
export PIP_CACHE_DIR="$SCRATCH/.cache/pip" XDG_CACHE_HOME="$SCRATCH/.cache" TMPDIR="$SCRATCH/.tmp" PYTHONNOUSERSITE=1
ml purge; ml load python/3.12.1; ml load cuda/12.8.0
source /home/users/ccitren/venvs/sing-py312/bin/activate

echo "=== node: $(hostname)  job: $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

for REP in 1 2 3; do
    echo "### gmix repeat $REP"
    python -u demos/bench_duffing_scaling.py --T 100000 --method efgp \
        --estep gmix --out-dir "demos/_bench_duffing_estep_rep/gmix_rep$REP"
    echo
done

for REP in 1 2; do
    echo "### analytic-o1 repeat $REP"
    python -u demos/bench_duffing_scaling.py --T 100000 --method efgp \
        --estep analytic --analytic-order 1 \
        --out-dir "demos/_bench_duffing_estep_rep/analytic_rep$REP"
    echo
done

echo "=== repeats done ==="
