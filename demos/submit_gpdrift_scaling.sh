#!/bin/bash
# Submit the 9-cell GP-drift scaling grid as independent GPU jobs.
#   methods : EFGP, SparseGP M=49 (7x7), SparseGP M=100 (10x10)
#   T       : 1000, 10000, 100000
# Each cell is its own job -> parallel across GPUs, incremental npz saves,
# and one cell's OOM/failure can't take down the others.
#
# Usage:
#   bash demos/submit_gpdrift_scaling.sh            # submit all 9 (seed 0)
#   SEED=1 bash demos/submit_gpdrift_scaling.sh     # another seed later
#   DRYRUN=1 bash demos/submit_gpdrift_scaling.sh   # print, don't submit
set -uo pipefail
cd /scratch/users/ccitren/sing-efgp
mkdir -p logs demos/_bench_duffing_scaling_out

SEED="${SEED:-0}"
LSINIT="${LSINIT:-0.7}"
OUTDIR="demos/_bench_duffing_scaling_out"
SBATCH_FILE="demos/gpdrift_scaling.sbatch"
# Pin every cell to the SAME GPU model (swl1 has both A100_SXM4 and H100_SXM5).
# Flip to A100 with:  GPUCON=GPU_SKU:A100_SXM4 bash demos/submit_gpdrift_scaling.sh
GPUCON="${GPUCON:-GPU_SKU:H100_SXM5}"

# Per-T resources. GPU memory (80GB, not host --mem) is the real ceiling for the
# big SparseGP cells. Host --mem is capped by swl1 MaxMemPerCPU=16G * (-c 4) = 64G,
# so keep requests <= 62G.
res_for_T() {
  case "$1" in
    1000)   echo "--mem=32G --time=00:40:00" ;;
    10000)  echo "--mem=48G --time=01:30:00" ;;
    100000) echo "--mem=62G --time=04:00:00" ;;
    *)      echo "--mem=48G --time=01:30:00" ;;
  esac
}

# method-config list: "METHOD M shortname"
CONFIGS=("efgp 0 efgp" "sp 49 sp49" "sp 100 sp100")
TS=(1000 10000 100000)

for T in "${TS[@]}"; do
  RES="$(res_for_T "$T")"
  for cfg in "${CONFIGS[@]}"; do
    read -r METHOD M SHORT <<<"$cfg"
    JOBNAME="gpd_${SHORT}_T${T}"
    CMD=(sbatch $RES -C "$GPUCON" -J "$JOBNAME"
         --output="logs/${JOBNAME}-%j.out" --error="logs/${JOBNAME}-%j.err"
         --export="ALL,T=${T},METHOD=${METHOD},M=${M},SEED=${SEED},LSINIT=${LSINIT},OUTDIR=${OUTDIR}"
         "$SBATCH_FILE")
    if [[ -n "${DRYRUN:-}" ]]; then
      echo "${CMD[*]}"
    else
      echo -n "submitting ${JOBNAME}: "
      "${CMD[@]}"
    fi
  done
done
