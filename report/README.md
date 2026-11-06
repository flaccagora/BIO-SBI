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

As of 6 November 2026 every sweep of E0–E4 is complete and has its figure, as do the
E2c rerun with prior-drawn truths (`fig_e2c_prior`, from `results/e2c_prior/*.csv`) and
X5, the role of the prior (`fig_x5a`–`fig_x5d`, from the CSVs in `results/x5/`), X2 seed
spread (`fig_x2`, from `results/x2/x2_spread.csv`), the E4 5k-vs-20k checks
(`fig_e4_budget`) and E0 at 4096 clones (`fig_e0_large`); only the never-run experiments
(X1, X3, X4, R, P0, follow-ups) are shown as an orange "Result pending" box. `make_figures.py` still checks that a sweep is complete before plotting
it (expected run counts are in each `fig_*` function) and skips it otherwise, so the
report shows a pending box (`\figorpending` in `report.tex`) until the sweep has
finished; the Comment text and the executive summary must then be updated by hand.

Numbers in the text come from the `info.md` analyses in `results/` (overview:
`results/overview.md`); the plotted numbers are in `tables/*.csv`.
