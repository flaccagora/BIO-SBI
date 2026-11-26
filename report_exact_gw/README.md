# Exact likelihood for the two-type Galton–Watson model — report

Standalone report on the exact GW likelihood (`inference=exact_gw`) and the XG2 experiment
(how many clones per endpoint the full transition model needs; 9 truths × 3 datasets ×
n = 128–1024 on generations 4/6/8/10; 108 NUTS posteriors plus the MLE of every run).

| File | Content |
|---|---|
| `report.tex` / `report.pdf` | the report: summary, model, method (likelihood, adjoint kernel, NUTS, point estimators), earlier exact-GW results (XG), XG2 design and results, why the posterior mean of B→A is biased, conclusions and follow-ups, reproducibility, glossary |
| `make_figures.py` | builds `figures/*.pdf` and `tables/*.{tex,csv}` from `results/exact_gw/data_size/` and copies four figures from `docs/xg2_mle_vs_posterior/figures/` |
| `figures/`, `tables/` | generated; `tables/summary.json` holds a few headline counts |

## Rebuild

From the repository root (results live in the main checkout's `results/`):

```bash
uv run python report_exact_gw/make_figures.py --results /path/to/BIO-SBI/results
cd report_exact_gw && /home/matteo/miniconda3/envs/latex/bin/tectonic report.tex
```

Inputs: `results/exact_gw/data_size/{xg2_runs.csv, xg2_mle.csv, runs_index.csv}` and each
run's `metrics.json` / `posterior_samples.csv.gz` (written by
`scripts/analyse_exact_gw_data_size.py` and `scripts/exact_gw_mle.py`). Some numbers in the
text (kernel speed, optimiser benchmark, earlier XG table, identifiability analysis) are
quoted from `scripts/gw_speedup/`, `scripts/pso_benchmark/results.md`,
`report_sequential/report.tex` and `docs/xg2_mle_vs_posterior/README.md`.
