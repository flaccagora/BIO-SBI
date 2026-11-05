# BIO-SBI

Simulation-based inference (SBI) for **two-type cell branching processes**: generate
synthetic clonal data under a chosen biological model and assay, fit a neural
posterior with [`sbi`](https://sbi-dev.github.io/sbi/), and measure how well the
parameters can be recovered. The goal is to **plan real experiments** (which endpoints,
how many clones, how much assay noise can be tolerated) before running them, and then
to fit the measured data with the same pipeline.

Every part of the pipeline is a swappable [Hydra](https://hydra.cc) config group:

| Group | Options | What it controls |
|---|---|---|
| `simulator` | `galton_watson`, `bellman_harris` | Generative model (time is generations or hours) |
| `law` | `reversible_ab`, `hierarchy_ab`, `minimal_ab` | Cell types, allowed fates, Dirichlet prior |
| `observation` | `perfect`, `noisy`, `all_clones` | Capture, type confusion, clone selection |
| `summary` | `moments`, `means` | Summary statistics fed to the network |
| `inference` | `npe`, `nle`, `nre` | sbi method family |
| `budget` | `smoke`, `dev`, `full` | Simulations, epochs, posterior/SBC sample sizes |
| `experiment` | `compare_methods`, `design_sweep`, `noise_sensitivity` | Ready-made multirun sweeps |

Each run saves its fully resolved config, so you can reproduce it from the run
directory alone.

---

## Install

You need [uv](https://docs.astral.sh/uv/). Python 3.12 is installed automatically.

```bash
git clone --recurse-submodules https://github.com/flaccagora/BIO-SBI.git
cd BIO-SBI
uv sync --group dev          # creates .venv from uv.lock
uv run pytest -q -m "not slow"   # ~5 s sanity check
```

## Quick start

```bash
uv run bio-sbi budget=smoke                    # Galton-Watson + NPE, a few seconds
uv run bio-sbi simulator=bellman_harris inference=nle
uv run bio-sbi law=hierarchy_ab observation=noisy design.clones=64 design.horizons=[6,12]
```

The results go to `outputs/<simulator>-<law>-<method>/<timestamp>/`:

| File | Contents |
|---|---|
| `.hydra/config.yaml`, `.hydra/overrides.yaml` | Full resolved config and the CLI overrides |
| `manifest.json` | Package versions, git commit/dirty flag, hardware |
| `reference.npz` | True parameters, simulated reference counts per horizon, summary `x` |
| `posterior.pt` | Trained posterior (`FittedPosterior`, reload with `FittedPosterior.load`) |
| `posterior_samples.csv.gz` | Posterior samples at the reference data (physical units) |
| `recovery.csv` | Per parameter: truth, mean, 90% CI, z-score, **contraction** |
| `metrics.json` | Everything above plus SBC calibration and timings |
| `sbc.npz`, `figures/sbc.png` | SBC ranks and rank histograms |
| `figures/posterior.png` | Prior vs posterior marginals with the true value |
| `tensorboard/` | sbi training curves (`uv run tensorboard --logdir outputs`) |

**Contraction** is `1 - Var_post / Var_prior`. A value near 0 means the design does not
inform that parameter, which is the main number to watch when planning an experiment.

## Running many experiments

Use Hydra multirun (`-m`). Comma-separated values form a grid, and every job gets its
own seeded directory under `multirun/<timestamp>/`.

```bash
# Methods x simulators x seeds (preset)
uv run bio-sbi -m +experiment=compare_methods

# Ad-hoc grid: how many clones / which endpoints are needed?
uv run bio-sbi -m design.clones=16,32,64,128 'design.horizons=[12],[4,8,12]' inference=npe

# Assay noise
uv run bio-sbi -m observation=noisy 'observation.capture=[0.9,0.9],[0.6,0.6]'

# Collect all metrics.json files into two CSV tables
uv run bio-sbi-collect multirun/<timestamp>
#   -> summary_runs.csv        one row per run (overrides + headline metrics)
#   -> summary_parameters.csv  one row per run x parameter (contraction, error, ...)
```

The `design_sweep` preset uses Galton-Watson horizons (generations). For Bellman-Harris,
pass horizons in hours.

## Components

### Simulators

Both models share the same **type graph** from the `law` group, so you can compare
discrete and continuous time under the same biological hypothesis.

* **Galton-Watson** (`simulator=galton_watson`): synchronous generations, where every
  cell has exactly one event per generation. Horizons are integer generations. With
  `selection: detected` it uses an **exact detection-conditioned sampler**, which prunes
  subtrees that leave no captured cells and so needs no rejection loop for rare
  survivors.
* **Bellman-Harris** (`simulator=bellman_harris`): an age-dependent continuous-time
  process. Each cell lives `Gamma(shape, mean/shape)` hours and then draws an outcome.
  The mean lifetimes per type are inferred with log-uniform priors
  (`lifetime_mean_bounds`), and `lifetime_shape` is fixed. `shape: 1` gives the Markov
  (exponential) process.

### Laws (type graphs and priors)

A law lists, for each type, its outcomes and their daughters `[n_A, n_B]`:

```yaml
# conf/law/reversible_ab.yaml (abridged)
outcomes:
  A: {death: [0,0], persist: [1,0], switch: [0,1], asymmetric: [1,1], self_renew: [2,0], sym_diff: [0,2]}
  B: {death: [0,0], persist: [0,1], switch: [1,0]}
concentration: null      # Dirichlet alphas per type; null = uniform simplex
reference: {A.death: 0.03, ...}   # truth for the reference dataset
founder: [1.0, 0.0]      # founder-type probabilities at seeding
```

* To fix an outcome's probability to zero, delete that outcome. To make the prior
  informative, raise its `concentration`.
* The parameters are named `"<type>.<outcome>"`, and Bellman-Harris adds
  `"<type>.mean_lifetime"`.
* `truth.values` overrides single reference values, e.g.
  `'truth.values={A.death: 0.2, A.self_renew: 0.38}'` (the rows must still sum to one).
  `truth.source=prior` draws the truth from the prior instead.

### Observation model

`capture` (per-type detection probability), `confusion` (row-stochastic labelling
matrix) and `selection` (`detected` keeps only clones with at least one detected cell;
`all` keeps every seeded clone). The error rates are treated as known.

### Inference methods

| Method | Learns | Posterior sampling | When to prefer |
|---|---|---|---|
| `npe` | `q(theta \| x)` (normalising flow) | direct, instant | default; cheap calibration checks; amortised over datasets |
| `nle` | `q(x \| theta)` (flow over summaries) | MCMC (slice) | likelihood reusable under other priors; robust sequential rounds |
| `nre` | `r(x, theta)` (classifier, NRE-C) | MCMC (slice) | simple and sample-efficient training |

`inference.method.rounds=R>1` switches to **sequential** inference at the reference
data (NPE-C/APT for NPE). This is more simulation-efficient for one dataset, but it
is not amortised, so SBC is skipped. The keys under `inference.method.network`,
`.training` and `.sampling` are passed straight to `sbi`.

> NLE and NRE sample with MCMC, so each SBC case costs one MCMC run. Keep
> `budget.sbc_cases` modest, or set it to 0, for these methods.

### Budgets

| Budget | Simulations | Max epochs | SBC cases |
|---|---:|---:|---:|
| `smoke` | 256 | 5 | 10 |
| `dev` | 5,000 | 200 | 200 |
| `full` | 50,000 | 500 | 1,000 |

`smoke` checks that the pipeline runs; it says nothing about accuracy.

## Fitting measured data

Train an amortised run whose design matches the experiment (same horizons, roughly
the same number of clones per horizon, same assay settings), then condition it on a
CSV with one row per clone:

```csv
horizon,A,B
12,35,10
12,4,0
...
```

```bash
uv run bio-sbi budget=full design.horizons=[12] design.clones=100 'law.founder=[0.7,0.3]'
uv run bio-sbi-infer outputs/<run> my_counts.csv --out results/sample1
```

This writes `posterior_samples.csv.gz`, `posterior_summary.csv` and `posterior.png`.
Horizons must match the trained design; a different clone count only produces a
warning. Before trusting a real-data fit, check calibration (SBC) for that design.

## Reproducibility

* **Seeds**: every stochastic stage uses a keyed random stream derived from `seed`
  (`bio_sbi.rng`). Results do not depend on `workers` or on simulation order, and
  identical configs give identical metrics (this is tested).
* **Config**: Hydra stores the composed config and the overrides in every run.
* **Environment**: `uv.lock` pins every dependency (`sbi==0.27.0`), and
  `manifest.json` records versions and the git commit.
* **Invalid simulations** (clones above `max_cells`, or recovery beyond the rejection
  budget) are dropped from training and their count is reported. The posterior is
  then implicitly conditioned on those limits, so set them well above any clone size
  you expect to observe.

## Extending

All components are instantiated with Hydra `_target_`. To add a new one, you write a
class and a YAML file; no registry needs editing.

| Add a... | Subclass | Implement | Config |
|---|---|---|---|
| simulator | `bio_sbi.simulators.BranchingSimulator` | `simulate()` (+ optional `simulate_detected`, `scalar_parameters`) | `conf/simulator/<name>.yaml` with `name`, `horizons`, `reference`, `model._target_` |
| inference method | `bio_sbi.inference.InferenceMethod` | `make_trainer()` (+ optional `append`, `build_posterior`) | `conf/inference/<name>.yaml` with `name`, `method._target_` |
| summary | `bio_sbi.summaries.Summary` | `__call__(cohorts) -> np.ndarray` | `conf/summary/<name>.yaml` |
| law | — | YAML only | `conf/law/<name>.yaml` |

From Python:

```python
from hydra import compose, initialize_config_module
from bio_sbi.pipeline import build_components, run_experiment

with initialize_config_module("bio_sbi.conf", version_base="1.3"):
    cfg = compose("config", overrides=["simulator=bellman_harris", "budget=smoke"])
comp = build_components(cfg)                # simulator, generator, method
theta = comp.simulator.prior().sample((8,)).numpy()
x, cohorts = comp.generator.simulate_theta(theta, stage="explore")
```

## Repository layout

```text
src/bio_sbi/
  simulators/     law.py (type graph, priors) · base.py · galton_watson.py · bellman_harris.py
  inference/      base.py (training loop, FittedPosterior) · methods.py (NPE, NLE, NRE)
  observation.py  capture / confusion / selection
  design.py       horizons x clones
  summaries.py    CohortMoments, CohortMeans
  simulate.py     DataGenerator: theta -> datasets -> x (threaded, keyed seeds)
  evaluation.py   recovery table, SBC + coverage
  pipeline.py     run_experiment, load_run, infer_measured
  cli.py          bio-sbi (Hydra) and bio-sbi-infer entry points
  collect.py      bio-sbi-collect
  conf/           Hydra config groups (see the table at the top)
tests/            unit tests (law, simulators vs theory, generator) + end-to-end runs
docs/models.md    model definitions, coordinates and glossary
submodules/GW     earlier Julia/Python reference work and organoid count data
```

## Development

```bash
uv run ruff check src tests && uv run ruff format --check src tests
uv run pytest -q -m "not slow"   # unit tests, ~5 s
uv run pytest -q                 # + end-to-end NPE/NLE/NRE runs, ~2 min
```

Version 0.2 is a rewrite focused on endpoint-count experiments. Earlier prototypes
(tracking observations, learned tree encoders, the identifiability playground, HTML
reports, the legacy Julia comparison) are not part of it.
