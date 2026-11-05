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

## Pending experiments

`make_figures.py` checks that a sweep is complete before plotting it (expected run
counts: E4 GW 120, E1c GW 80, E1a founder number GW 30, E2c model-C arms 30 each,
E2a/E2b GW 30 each in `results/e2/known*gw*` / `ignored*gw*`). Incomplete sweeps are
skipped and the report shows an orange "Result pending" box instead. Once a sweep has
finished, re-run the script and recompile: the figure replaces the box automatically
(`\figorpending` in `report.tex`). The Comment text and the executive summary must then
be updated by hand.

Numbers in the text come from the `info.md` analyses in `results/` and, for the GW
E1a founder-type and E2c ignore-C sweeps (not analysed there), from `tables/*.csv`.
