# Models, coordinates and terminology

## Data hierarchy

| Term | Meaning |
|---|---|
| **Founder** | A seeded cell. A clone has `N ~ founder_size` founders with i.i.d. types `~ founder`. |
| **Clone** | All descendants of one seeding event (one or more founders). |
| **Cohort** | Independent clones measured at one common endpoint (*horizon*). |
| **Dataset** | Several cohorts at different horizons, sharing one parameter vector. |
| **Design** | `design.horizons` × `design.clones`: what an experiment measures. |
| **Recovered clone** | A clone with ≥ 1 detected cell at its endpoint (`selection: detected`). |

Endpoint assays are destructive, so different horizons always use different clones.

## Offspring law (shared by both simulators)

Each type `k` has outcomes `o = 1..O_k`, each with a probability `p_{k,o}` and a
daughter vector `d_{k,o}` (one count per type). One *event* replaces a cell by its
daughters. With `outcomes: all` the outcomes are death, `persist`, `to_<T>` for every
other type, and every unordered division `div_<T1><T2>` (`max_daughters: 2`), which
is `1 + K + K(K+1)/2` outcomes for `K` types. `persist` and `to_<T>` keep a single
cell and create no lineage branch.

**Founders.** A clone is seeded with `N` cells, `P(N = n) = founder_size.n` for
`n ∈ founder_sizes`, and each founder's type is i.i.d. `founder.<k>`. Founders
contribute independent sub-clones, which are summed into one measured clone.

## Parameter space

All probabilities sit in simplex rows: one per type, plus `founder` and
`founder_size`. Simulator scalars (BH lifetimes) are appended.

**Fixed values.** `law.fixed` (and `simulator.model.fixed` for scalars) pins
parameters. In a row with fixed mass `F`, the free entries are
`(1 - F) · Dirichlet(alpha_free)`. If one free entry is left it is determined (no
coordinate), and if `F = 1` the remaining entries are zero.

**Prior.** The free entries of each row are `Dirichlet(alpha)` (`law.concentration`,
default 1). Free scalars are log-uniform on their bounds.

**Inference coordinates.** Each free row is parameterised by stick-breaking fractions
`v_i ~ Beta(alpha_i, sum_{j>i} alpha_j)`, which are independent, and
`p_i = (1 - F) v_i prod_{j<i}(1 - v_j)`. This is an exact reparameterisation of the
Dirichlet with a box-shaped support, so NPE, NLE and NRE (whose MCMC transforms to
unconstrained space) all use it directly. `theta_names` are `"<row>.stick_<entry>"`.

**Mean matrix.** `M[k, j] = sum_o p_{k,o} d_{k,o,j}`, which is the expected number of
type-`j` daughters per type-`k` event.

## Galton-Watson (`simulator=galton_watson`)

Synchronous generations: every living cell has one event per generation and its
daughters act in the next one. For `Z_g` the counts after `g` generations,
`E[Z_G] = E[N] · founder · M^G` (checked in the tests).

**Detection-conditioned sampling.** Let `s_g(k)` be the probability that a type-`k`
cell with `g` generations left has at least one *captured* descendant:

```
s_0(k) = capture[k]
s_g(k) = sum_o p_{k,o} (1 - prod_j (1 - s_{g-1}(j))^{d_{k,o,j}})
```

A recovered clone is sampled top-down:

1. each founder succeeds with `q = sum_k founder[k] s_G(k)`: draw `N` with weights
   `P(N = n)(1 - (1 - q)^n)`, then the founders' success flags conditional on at least
   one success, then each successful founder's type with weights `founder[k] s_G(k)`;
2. at each cell, draw the outcome with weights `p_{k,o}(1 - prod_j(...))`;
3. draw the daughters' success flags conditional on at least one success;
4. recurse only into the daughters that succeed.

This draws exactly from the captured counts given detection, at a cost proportional to
the size of the observed tree. The founder mixture among recovered clones is therefore
reweighted by detectability, and generally differs from `founder` and `founder_size`.

**Detection probability.** A seeded clone is recovered with probability
`sum_n P(N = n)(1 - (1 - q)^n)` (`GaltonWatson.detection_probability`). This is the
exact *observed-organoid fraction*, reported per horizon for the reference dataset as
`reference_detection[*].detection_probability` in `metrics.json`. The exact sampler
never simulates undetected clones, so it has no empirical `detection_rate`.

## Bellman-Harris (`simulator=bellman_harris`)

This is an age-dependent process. A type-`k` cell born at time `t` lives
`L ~ Gamma(shape_k, mean_k / shape_k)` (mean `mean_k`, CV `1/sqrt(shape_k)`) and then
draws an outcome, and its daughters are born at `t + L`. The counts are of the cells
alive at the horizon `T`. All founders are born at `t = 0`.

* Physical parameters: the outcome probabilities plus `"<type>.mean_lifetime"`. The
  latter are inferred as `log mean ~ Uniform(log low, log high)`.
* `shape = 1`: Markov process with `E[Z_T] = E[N] · founder · exp(T · diag(1/mean)(M - I))`
  (checked in the tests).
* `shape → ∞`: deterministic lifetimes, which reduce to Galton-Watson with generation
  time `mean` (checked in the tests).
* Clones under `selection: detected` are recovered by forward simulation plus rejection
  (`observation.max_attempts_per_clone`). For the reference dataset the rejection loop
  reports `seeded` (clones simulated), `detected` (those with >= 1 detected cell) and
  `detection_rate = detected / seeded`, an estimate of the observed-organoid fraction.

## Observation model

The order of operations is biology, then capture `Binomial(n_k, capture[k])` per
*true* type `k`, then labelling `Multinomial(·, confusion[k])` over the measured
labels, then selection. `confusion` is `true types × labels`, so it may be rectangular:
three true types read through two markers is a `3 × 2` matrix. An undetected endpoint
cell is not a death, and an undetected clone is not an extinct clone.
`observation=three_split` is such a matrix with a hidden C read as A with probability
`q` and as B with `1 - q`.

## Data model vs inference model

`data.simulator`, `data.law` and `data.observation` generate the reference dataset;
`simulator`, `law` and `observation` define the prior, the training simulations and
calibration. The two must share the time unit and the label set. Everything else
(number of true types, transitions, founders, fixed values, assay) may differ.

The two sides also have separate root seeds: `data_seed` (default `seed`) drives the
truth draw and the reference dataset (stream keys `"truth"` and `"reference"`); `seed`
drives everything on the inference side (`"round-<r>"` training simulations, network
training, `"ref"` posterior sampling, `"prior-reference"` prior samples for
contraction, `"ppc"`, `"sbc-theta"`/`"sbc"`, and `"measured"` in `bio-sbi-infer`).

* **Recovery** compares posterior and truth by parameter name. A name missing from
  the data model has `truth = NaN` and is not scored. Names that match can still
  mean different things under misspecification: e.g. inference-"B" absorbs a hidden C.
* **Posterior predictive check.** For draws `theta_s` from the posterior, simulate
  `x_s` under the inference model. For each summary feature, compute the two-sided
  tail probability of the observed value, `p = 2 min(F, 1 - F)`.
  `fraction_outside_95` is the share of features with `p < 0.05`. It is near 0.05 or
  below for an adequate model, and larger when the model cannot reproduce the data.
  Features are correlated and the check uses the data twice, so treat it as a relative
  diagnostic: compare it with the well-specified run at the same design.

## Summaries

`CohortMoments` computes, per cohort, the mean, standard deviation and quantiles of
`log1p` counts per type and of the total, the fraction of clones containing each type,
the fraction of empty clones, and the correlation of log counts between types. Features
that are constant over the first round of training simulations are masked out (this
matters for NLE), and the same mask is applied to the observed data.

## Calibration

SBC is run in `theta` (stick) coordinates via `sbi.diagnostics.run_sbc`. For each
coordinate, the normalised rank `u = (rank + 0.5)/L` should be uniform. The KS p-values
and the empirical coverage of central credible intervals at 50/80/90/95% are reported.
Recovery statistics at the reference truth are reported in physical units.
