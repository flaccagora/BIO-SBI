# BIO-SBI

Simulation-based inference (SBI) for **multi-type cell branching processes**: generate
synthetic clonal data under a chosen biological model and assay, fit a neural
posterior with [`sbi`](https://sbi-dev.github.io/sbi/), and measure how well the
parameters can be recovered. The goal is to **plan real experiments** (which endpoints,
how many clones, how much assay noise can be tolerated) before running them, to
**test what happens when a modelling assumption is wrong** (hidden cell types,
transitions fixed to the wrong value, multi-cell seeding), and then to fit the
measured data with the same pipeline.

Every part of the pipeline is a swappable [Hydra](https://hydra.cc) config group:

| Group | Options | What it controls |
|---|---|---|
| `simulator` | `galton_watson`, `bellman_harris` | Generative model (time is generations or hours) |
| `law` | `two_type` (all transitions), `reversible_ab`, `hierarchy_ab`, `minimal_ab`, `three_type`, `three_type_traj`, `multi_founder` | Cell types, transitions, founders, fixed values, prior |
| `observation` | `perfect`, `noisy`, `all_clones`, `three_as_two`, `three_split` | Capture, type→label confusion, clone selection |
| `summary` | `moments`, `means` | Summary statistics fed to the network |
| `inference` | `npe`, `nle`, `nre` | sbi method family |
| `budget` | `smoke`, `dev`, `full` | Simulations, epochs, posterior/SBC sample sizes |
| `design` | `default`, `real_like`, `single_128`, `three_128`, `single_384` | Endpoints and clones per endpoint |
| `experiment` | `e1a_*` … `e4_design`, `x2_seeds` (see [TODO.md](TODO.md)), `b_row_*`, `design_comparison`, `compare_methods`, `design_sweep`, `noise_sensitivity` | Ready-made multirun sweeps (each documents itself in `info.md`) |

`simulator`, `law` and `observation` describe the **inference model** (what you assume).
The reference data come from a separate **data model**, `data.simulator`, `data.law`
and `data.observation`. It mirrors the inference model unless you override it (see
[Testing assumptions](#testing-assumptions-misspecification)). Each run saves its fully
resolved config, so you can reproduce it from the run directory alone.

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
| `recovery.csv` | Per parameter: truth, mean, 90% CI, z-score, **contraction**, inferred/fixed, `at_boundary` (truth 0 or 1: no coverage score) |
| `metrics.json` | Everything above plus SBC calibration, timings, `seed`/`data_seed` and `reference_detection` (observed-organoid fraction per horizon, see below) |
| `ppc.npz` | Posterior-predictive summaries (misspecification check) |
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

# Group by sweep folder + swept keys and average over seeds
uv run python scripts/summarize.py results/<exp> [--params 'A.*' B.to_A] [--group-by KEY ...]
#   -> summary_groups.csv            per group: runs, error, contraction, coverage, SBC, PPC
#   -> summary_group_parameters.csv  per group x parameter: truth, mean, sd, error,
#                                    contraction, 90% coverage across seeds, recovered
```

The experiment presets `e1a_*` to `e4_design` and `x2_seeds` implement the sweeps of
the experiments plan ([TODO.md](TODO.md)); each file's header gives the exact commands
(some need one run per inference arm, with a separate `hydra.sweep.dir`) and the run
count. They use the baseline of the plan (`law=reversible_ab`, `design=real_like`:
days 6 and 10 with 165 and 101 clones, in generations or hours depending on the
simulator).

**Observed-organoid fraction.** For the reference dataset, `metrics.json` lists per
horizon `seeded` and `detected` clones and `detection_rate = detected / seeded` (only
when clones are recovered by forward simulation plus rejection, e.g. Bellman-Harris)
and `detection_probability`, the exact P(clone has >= 1 captured cell) for
Galton-Watson. `bio-sbi-collect` adds them as `detection_rate_h<h>` /
`detection_probability_h<h>` columns. Training simulations do not record them.

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

### Laws: types, transitions, founders, fixed values

With `outcomes: all` (the default) **every transition is enabled**. For types `A, B`,
each type has these outcomes:

| Outcome | Daughters of an A cell | Parameter |
|---|---|---|
| `death` | none | `A.death` |
| `persist` | A (no division) | `A.persist` |
| `to_B` | B (single-cell switch) | `A.to_B` |
| `div_AA`, `div_AB`, `div_BB` | two cells | `A.div_AA`, ... |

B has the same outcomes (`B.to_A`, `B.div_AB`, ...), and `three_type` adds C (10
outcomes per type). A clone starts from **N founder cells**, with
`P(N = n) = founder_size.<n>` over `founder_sizes`, and each founder independently has
type `k` with probability `founder.<k>`. All of these are ordinary parameters.

```yaml
# conf/law/two_type.yaml (abridged)
types: [A, B]
outcomes: all            # or an explicit {type: {name: [n_A, n_B]}} table
founder_sizes: [1]       # e.g. [1, 2, 3] when single-cell seeding can fail
fixed:                   # not inferred, and used when this law generates data
  founder: {A: 1.0}
concentration: {}        # Dirichlet alphas, e.g. {founder_size: {"1": 8.0}}
reference:               # ground truth when this law generates the reference data
  A: {death: 0.03, persist: 0.03, to_B: 0.05, div_AB: 0.05, div_AA: 0.55, div_BB: 0.29}
  B: {death: 0.10, persist: 0.10, to_A: 0.80}   # omitted entries are 0
```

* **Fix any parameter** to a value (0 disables a transition) and it is no longer
  inferred. The free entries of a row share the leftover mass `1 - sum(fixed)` under a
  Dirichlet prior. From the command line, use the nested form: `+` adds a key and
  `++` replaces a whole mapping.

  ```bash
  uv run bio-sbi +law.fixed.B.div_AA=0 +law.fixed.B.div_AB=0 +law.fixed.B.div_BB=0
  uv run bio-sbi +law.fixed.A.persist=0.03    # fix to the ground-truth value
  uv run bio-sbi '++law.fixed.founder={A: 0.7, B: 0.3}'
  ```

> **Command-line edits to `law.*`, `simulator.*` and `observation.*` change only the
> inference model.** The data model selects the same YAML *option* but keeps its own
> values, so `+law.fixed.A.persist=0.2` means "data have the true value, the model
> assumes 0.2". To change both, override both, e.g.
> `law.founder_sizes=[1,2] data.law.founder_sizes=[1,2]`, or write a new law file.
> `metrics.json` records `well_specified: true` only when both sides are identical.

  `reversible_ab`, `hierarchy_ab` and `minimal_ab` are just `two_type` with some
  transitions fixed to 0.
* **Multi-cell founders**: `law=multi_founder` seeds 1 to 3 cells of mixed type and
  infers both the size and the type distribution. You can also set
  `'law.founder_sizes=[1,2]'` together with `+law.fixed.founder_size.2=0.1` to fix the
  doublet rate.
* `truth.values` overrides single reference values (rows must still sum to one), e.g.
  `'truth.values={A: {death: 0.2, div_AA: 0.38}}'`. `truth.source=prior` draws the
  truth from the prior instead.
* Bellman-Harris adds `<type>.mean_lifetime`, which you can fix with
  `+simulator.model.fixed.B.mean_lifetime=30`.

### Observation model

* `capture`: detection probability per *true* type (one value, a list or
  `{type: p}`).
* `confusion`: the probability that a true type receives a measured label. It is
  rectangular when there are more true types than markers, e.g. `{C: {B: 1}}` reads
  a hidden C as B (`observation=three_as_two`); `observation=three_split` reads C as
  A with probability `q` and as B with `1 - q` (`observation.q=0.3`, a top-level key
  of that file interpolated into the matrix). A single number is a symmetric
  mislabelling rate.
* `selection`: `detected` keeps only clones with at least one detected cell; `all`
  keeps every seeded clone.

The error rates are treated as known.

### Testing assumptions (misspecification)

The data model is configured separately from the inference model through
packaged groups (`<group>@data.<group>=<option>`) or `data.*` keys:

```bash
# Data contain a hidden third type (C, read as B); inference assumes two types
uv run bio-sbi law@data.law=three_type observation@data.observation=three_as_two

# Data have all transitions; inference assumes a strict hierarchy
uv run bio-sbi law=hierarchy_ab

# Data come from 1-3 founder cells; inference assumes single-cell seeding
uv run bio-sbi law@data.law=multi_founder

# Inference fixes a transition to a wrong value
uv run bio-sbi +law.fixed.A.persist=0.2

# Continuous-time data with a two-type BH inference model, three types in the data
uv run bio-sbi simulator=bellman_harris law@data.law=three_type \
  observation@data.observation=three_as_two
```

Data and inference models must use the same time unit and produce the same
measured labels; the number of true types, the transitions and the parameters may
differ. In `recovery.csv`, the truth is matched **by parameter name**; a parameter the
data model does not have gets `truth = NaN`. Look at the parameters' meaning before
reading their errors: under a hidden-type misspecification, "B" in the inference model
absorbs B and C. `metrics.json` records `well_specified` and the
**posterior-predictive check** (`ppc`), which reports the fraction of summary features
the fitted model cannot reproduce (`fraction_outside_95`). That check works even when
the parameters are not comparable.

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
CSV with one row per clone and one column per measured label:

```csv
horizon,A,B
12,35,10
12,4,0
...
```

```bash
uv run bio-sbi budget=full design.horizons=[12] design.clones=100 \
  '++law.fixed.founder={A: 0.7, B: 0.3}'
uv run bio-sbi-infer outputs/<run> my_counts.csv --out results/sample1
```

This writes `posterior_samples.csv.gz`, `posterior_summary.csv` and `posterior.png`.
Horizons must match the trained design; a different clone count only produces a
warning. Before trusting a real-data fit, check calibration (SBC) for that design.

## Reproducibility

* **Seeds**: every stochastic stage uses a keyed random stream derived from a root
  seed (`bio_sbi.rng`). Results do not depend on `workers` or on simulation order, and
  identical configs give identical metrics (this is tested).
* **Data seed vs inference seed**: `data_seed` (default `${seed}`) seeds the data side
  only, the truth draw (`truth.source=prior`) and the reference dataset; `seed` seeds
  training simulations, network training, posterior sampling, prior samples for
  contraction, the PPC and SBC. `data_seed=3 seed=0,1,2,3,4` therefore trains five
  times on the same `reference.npz`. With the default, results equal those of earlier
  versions.
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
| law | — | YAML only (`outcomes: all` or an explicit table) | `conf/law/<name>.yaml` |

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
  simulators/     law.py (outcomes, founders, ParameterSpace: fixed/free, priors)
                  base.py · galton_watson.py · bellman_harris.py
  inference/      base.py (training loop, FittedPosterior) · methods.py (NPE, NLE, NRE)
  observation.py  capture / type->label confusion / selection
  design.py       horizons x clones
  summaries.py    CohortMoments, CohortMeans
  simulate.py     DataGenerator: theta -> datasets -> x (threaded, keyed seeds)
  evaluation.py   recovery table, posterior-predictive check, SBC + coverage
  pipeline.py     data vs inference model, run_experiment, load_run, infer_measured
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
