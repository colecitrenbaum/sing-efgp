#!/bin/bash
# Submit the 9-cell Duffing K-sweep (number of trials) as independent GPU jobs.
#   methods : EFGP, SparseGP M=49 (7x7), SparseGP M=400 (20x20)
#   K       : 1, 10, 100   (each trial T=1000)
# One job per cell -> parallel, incremental npz, isolated failures/OOM.
#
#   bash demos/submit_duffing_kscaling.sh          # all 9 (seed 0)
#   DRYRUN=1 bash demos/submit_duffing_kscaling.sh # print only
set -uo pipefail
cd /scratch/users/ccitren/sing-efgp
mkdir -p logs demos/_bench_duffing_kscaling_out

SEED="${SEED:-0}"
OUTDIR="demos/_bench_duffing_kscaling_out"
SBATCH_FILE="demos/duffing_kscaling.sbatch"
GPUCON="${GPUCON:-GPU_SKU:H100_SXM5}"

# Per-K resources (host --mem <= 62G: swl1 MaxMemPerCPU=16G * -c 4). GPU mem is
# the real ceiling; M=400 at K=100 is the OOM-risk cell (recorded as failed).
res_for_K() {
  case "$1" in
    1)   echo "--mem=32G --time=00:40:00" ;;
    10)  echo "--mem=48G --time=01:30:00" ;;
    100) echo "--mem=62G --time=04:00:00" ;;
    *)   echo "--mem=48G --time=01:30:00" ;;
  esac
}

CONFIGS=("efgp 0 efgp" "sp 49 sp49" "sp 400 sp400")
KS=(1 10 100)

for K in "${KS[@]}"; do
  RES="$(res_for_K "$K")"
  for cfg in "${CONFIGS[@]}"; do
    read -r METHOD M SHORT <<<"$cfg"
    JOBNAME="duffk_${SHORT}_K${K}"
    CMD=(sbatch $RES -C "$GPUCON" -J "$JOBNAME"
         --output="logs/${JOBNAME}-%j.out" --error="logs/${JOBNAME}-%j.err"
         --export="ALL,K=${K},T=1000,METHOD=${METHOD},M=${M},SEED=${SEED},EPS=0.001,OUTDIR=${OUTDIR}"
         "$SBATCH_FILE")
    if [[ -n "${DRYRUN:-}" ]]; then
      echo "${CMD[*]}"
    else
      echo -n "submitting ${JOBNAME}: "
      "${CMD[@]}"
    fi
  done
done
