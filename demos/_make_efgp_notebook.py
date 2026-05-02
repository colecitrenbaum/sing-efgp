"""Convert demos/efgp_synthetic_2d.py into a Jupyter notebook with section
breaks at every "# ----" header, plus a markdown intro cell at the top.

Run via:  python demos/_make_efgp_notebook.py
"""
from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf


SCRIPT = Path(__file__).resolve().parent / "efgp_synthetic_2d.py"
NOTEBOOK = Path(__file__).resolve().parent / "efgp_synthetic_2d.ipynb"

INTRO = """\
# EFGP-SING demo: latent SDE inference with a Fourier-feature GP drift

This notebook fits an EFGP-based GP-drift block to a small d=2 synthetic
latent SDE and compares against SING's stock inducing-point baseline.

* GP drift posterior, kernel hyperparameters and posterior-predictive moments
  are computed by `sing.efgp_drift.EFGPDrift` (Toeplitz / NUFFT / CG).
* The latent posterior `q(x)` is updated by SING's natural-gradient routine
  with the **local-quadratic transition approximation** wired in via
  `jax.custom_vjp`.

See `EFGP_SING_README.md` for the algorithm map and known limitations.
"""


def main():
    src = SCRIPT.read_text()
    # Split on commented section breaks.
    parts = re.split(r"\n# -{60,}\n", src)
    # Drop the leading shebang/imports cell and split-out hunks
    cells = [nbf.v4.new_markdown_cell(INTRO)]
    for chunk in parts:
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        # Try to extract a leading docstring or comment as a markdown cell
        m = re.match(r'#\s*(\d+\.\s.*)', chunk)
        if m:
            md = "## " + m.group(1)
            cells.append(nbf.v4.new_markdown_cell(md))
        cells.append(nbf.v4.new_code_cell(chunk))

    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {
            "display_name": "Python 3 (myenv)",
            "language": "python",
            "name": "python3",
        }
    })
    NOTEBOOK.write_text(nbf.writes(nb))
    print(f"wrote {NOTEBOOK}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
