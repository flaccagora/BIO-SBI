# BIO-SBI simulation report, sequential edition

Same experiments, figures, tables and numbers as `../report/` (ordered by experiment ID),
but presented in the order of the argument: each experiment appears where an earlier
result motivates it, with a "Why this next" box citing that result and a "Leads to" box
pointing forward.

| File | Content |
|---|---|
| `report.tex` | the report: executive summary (in narrative order), roadmap (dependency graph + experiment index with section numbers here and in `report/report.pdf`), methods, Parts I–VI, appendices (device benchmark, P0 timing, reproducibility, glossary) |
| `report.pdf` | compiled report |

Order: methodological checks (X2 seed spread, X1 inference methods) → I identifiability and the prior (E0, X5a, X5c, E0 at
4096 clones) → II design and amount of data
(design comparison, E4 GW, E4 BH) → III BH timing (BH ridge, X5d, X5b, E3a, E3b, E3c, X3 GW
fitted to BH data) → IV misspecified biology and assay (E1a type, E1a number BH/GW, E1b,
E1b follow-up, E1c BH/GW, E2a/b BH/GW, E2 asymmetric follow-up, E2c, E2c modelled, E2c rerun,
detectability synthesis, X4 combined errors) → V real data (CRC0322: data, model ladder, final
model) → VI planned follow-ups (real-data model, Q1–Q5, measurements). Appendix: device benchmark, P0 timing,
reproducibility, glossary.

## Dependency on `../report/`

This folder has **no figures of its own**. `report.tex` sets
`\graphicspath{{../report/figures/}}` and checks `\IfFileExists{../report/figures/<name>.pdf}`,
so it needs `report/figures/*.pdf` (written by `report/make_figures.py`). A missing figure
is shown as an orange pending box. The text and tables were taken from
`report/report.tex`; if that report is updated, the matching section here must be updated
by hand (sections keep the same labels, e.g. `sec:e0`, `sec:x5b`). The "ID-ord." column of
the experiment index (Table 1) holds the section numbers of `report/report.pdf` as literal
text: update it when sections are added there.

## Rebuild

From the repository root:

```bash
uv run python report/make_figures.py          # figures used by both reports
cd report_sequential && /home/matteo/miniconda3/envs/latex/bin/tectonic report.tex
```

Build from inside `report_sequential/` (the figure path is relative).
