# BIO-SBI simulation report

LaTeX report of all simulation experiments (synthetic data), for planning the wet-lab
organoid experiments.

| File | Content |
|---|---|
| `report.tex` | the report (executive summary, methods, one section per experiment, pending experiments, appendix) |
| `make_figures.py` | builds every figure from `results/` (read only) |
| `figures/*.pdf` | vector figures written by `make_figures.py` |
| `tables/*.csv` | the numbers behind the figures (shifts, grids, detection rates) |
| `report.pdf` | compiled report |

## Rebuild

From the repository root:

```bash
uv run python report/make_figures.py          # all figures
uv run python report/make_figures.py fig_e4   # only some (function names)
cd report && /home/matteo/miniconda3/envs/latex/bin/tectonic report.tex
```

Tectonic downloads the LaTeX packages on first use (no system LaTeX needed).

## Status and pending boxes

As of 6 November 2026 every sweep of E0–E4 is complete and has its figure; only X2 and
the never-run experiments (X1, X3, X4, R, P0, follow-ups) are shown as orange "Result
pending" boxes. `make_figures.py` still checks that a sweep is complete before plotting
it (expected run counts are in each `fig_*` function) and skips it otherwise, so the
report shows a pending box (`\figorpending` in `report.tex`) until the sweep has
finished; the Comment text and the executive summary must then be updated by hand.

Numbers in the text come from the `info.md` analyses in `results/` (overview:
`results/overview.md`); the plotted numbers are in `tables/*.csv`.
