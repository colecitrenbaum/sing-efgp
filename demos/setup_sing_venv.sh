#!/bin/bash
# Build a GPU-capable venv for SING/EFGP on Sherlock.
# Run on a compute node with a GPU (e.g. via: srun -p dev -G 1 -c 4 --mem 16G -t 01:00:00 bash demos/setup_sing_venv.sh)
set -uo pipefail

VENV=/home/users/ccitren/venvs/sing-py312
REPO=/scratch/users/ccitren/sing-efgp

# Keep caches off $HOME (15GB NFS quota).
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export TMPDIR="$SCRATCH/.tmp"
mkdir -p "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"

# CRITICAL: the user's ~/.local/lib/python3.12 (ipykernel/numpy/... from Jupyter)
# would otherwise leak into the venv. Isolate hard.
export PYTHONNOUSERSITE=1

echo "=== modules ==="
ml purge
ml load python/3.12.1
ml load cuda/12.8.0
ml load cmake/3.31.4
which python nvcc cmake
nvcc --version | tail -2

echo "=== create venv $VENV ==="
# NOTE: the python/3.12.1 module exposes `python3`/`python3.12` but leaves
# `python` -> system 2.7.5, so use python3 explicitly to build the venv.
if [ ! -x "$VENV/bin/python" ]; then
    rm -rf "$VENV"
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
python --version           # now the venv's 3.12
pip install --upgrade pip setuptools wheel

# Sherlock is RHEL7 / glibc 2.17: newer numpy/scipy ship only manylinux_2_28
# wheels (glibc 2.28), so pip slow-builds them from sdist (scipy even fails:
# "OpenBLAS not found"). Pin to the last versions with manylinux2014 wheels:
#   numpy 1.26.4 (matches the working rrl-py312 venv), scipy 1.13.1.
# jax 0.5.3 accepts numpy>=1.25, scipy>=1.11.1. Install all together so the
# resolver honours the pins.
echo "=== install jax[cuda12]==0.5.3 + pinned numpy/scipy (manylinux2014 wheels) ==="
pip install "jax[cuda12]==0.5.3" "numpy==1.26.4" "scipy==1.13.1"

echo "=== jax GPU check ==="
python - <<'PY'
import jax
print("jax", jax.__version__)
print("devices:", jax.devices())
PY

echo "=== install sing deps as wheels-only (glibc-2.17 safe), then sing --no-deps ==="
# --only-binary=:all: makes pip pick the newest version of each native dep
# (pillow via matplotlib, tensorstore via flax, ...) that actually has a
# manylinux2014 / glibc-2.17 wheel, instead of falling back to an sdist that
# won't build on RHEL7. jax/jaxlib pinned so flax can't drag in a newer jax
# that breaks the CUDA plugin match; flax may downgrade freely to suit.
pip install --only-binary=:all: \
    "jax==0.5.3" "jaxlib==0.5.3" "numpy==1.26.4" "scipy==1.13.1" \
    matplotlib flax optax tensorflow-probability
# sing is pure-python; install it without re-resolving deps (already satisfied).
pip install -e "$REPO" --no-deps

echo "=== build jax_finufft==1.0.1 with CUDA (cuFINUFFT) ==="
# Pin 1.0.1: build-requires only nanobind + scikit-build-core (NO jax build pin,
# unlike 1.3.x which build-pins jax==0.8.0 -> needs jaxlib 0.8.0 with no glibc-2.17
# wheel). Runtime dep jax>=0.5.0, so compatible with the installed jaxlib 0.5.3.
# It vendors the XLA FFI header, so no jaxlib headers are needed at build time.
export CUDA_HOME="${CUDA_HOME:-$(dirname $(dirname $(which nvcc)))}"
echo "CUDA_HOME=$CUDA_HOME  nvcc=$(which nvcc)"
# native arch is the dev-node GPU; set an explicit arch list so the cubin also
# runs on swl1 GPUs (V100=70, A100=80, H100=90); PTX for 90 forward-JITs beyond.
# jax-finufft 1.0.1's CMake hard-queries the `fftw3` target, so FFTW is required
# (DUCC0 doesn't create that target). The vendored FINUFFT FetchContent-downloads
# FFTW but leaves the version empty -> URL 'fftw-.tar.gz' 404s. Set FFTW_VERSION
# so it builds FFTW from source and defines the fftw3 target. GPU path uses cuFFT.
pip install --no-binary=jax-finufft "jax-finufft==1.0.1" \
    --config-settings=cmake.define.JAX_FINUFFT_USE_CUDA=ON \
    --config-settings=cmake.define.FFTW_VERSION=3.3.10 \
    --config-settings="cmake.define.CMAKE_CUDA_ARCHITECTURES=70;80;90" \
    -v
JF_RC=$?
echo "jax_finufft build rc=$JF_RC"

echo "=== verification ==="
python - <<'PY'
import numpy as np
import jax, jax.numpy as jnp
print("jax", jax.__version__, "devices:", jax.devices())
import jax_finufft
print("jax_finufft imported OK")
# Type-2 NUFFT: evaluate a small spectral grid at random points.
n = 8
coeffs = jnp.ones((n,), dtype=jnp.complex128)
pts = jnp.linspace(-1.0, 1.0, 5)
out = jax_finufft.nufft2(coeffs, pts)
print("nufft2 out device:", out.device if hasattr(out, "device") else "n/a",
      "dtype:", out.dtype, "shape:", out.shape)
print("nufft2 finite:", bool(np.all(np.isfinite(np.asarray(out)))))
import sing.efgp_em  # noqa: F401
print("sing.efgp_em imported OK")
PY
echo "=== DONE ==="
