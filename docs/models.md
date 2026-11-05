# Models, coordinates and terminology

## Data hierarchy

| Term | Meaning |
|---|---|
| **Founder** | Single seeded cell that starts a clone. Its type is drawn from `law.founder`. |
| **Clone** | All descendants of one founder. |
| **Cohort** | Independent clones measured at one common endpoint (*horizon*). |
| **Dataset** | Several cohorts at different horizons, sharing one parameter vector. |
| **Design** | `design.horizons` × `design.clones`: what an experiment measures. |
| **Recovered clone** | A clone with ≥ 1 detected cell at its endpoint (`selection: detected`). |

Endpoint assays are destructive, so different horizons always use different clones.

## Offspring law (shared by both simulators)

Each type `k` has outcomes `o = 1..O_k`, each with a probability `p_{k,o}` and a
daughter vector `d_{k,o} = (n_A, n_B)`. One *event* replaces a cell by its daughters.
`persist` (`[1,0]` for A) and `switch` (`[0,1]` for A) keep a single cell, so they create
no new lineage branch.

**Prior.** Each row is `p_k ~ Dirichlet(alpha_k)` (`law.concentration`, default all ones).

**Inference coordinates.** Each row is parameterised by stick-breaking fractions
`v_{k,i} ~ Beta(alpha_{k,i}, sum_{j>i} alpha_{k,j})`, which are independent, and
`p_{k,i} = v_{k,i} prod_{j<i}(1 - v_{k,j})`. This is an exact reparameterisation of
the Dirichlet with a box-shaped support, so NPE, NLE and NRE (whose MCMC transforms to
unconstrained space) all use it directly. `theta_names` are `"<type>.stick_<outcome>"`.

**Mean matrix.** `M[k, j] = sum_o p_{k,o} d_{k,o,j}`, which is the expected number of
type-`j` daughters per type-`k` event.

## Galton-Watson (`simulator=galton_watson`)

Synchronous generations: every living cell has one event per generation and its
daughters act in the next one. For `Z_g` the counts after `g` generations,
`E[Z_G] = founder · M^G` (checked in the tests).

**Detection-conditioned sampling.** Let `s_g(k)` be the probability that a type-`k`
cell with `g` generations left has at least one *captured* descendant:

```
s_0(k) = capture[k]
s_g(k) = sum_o p_{k,o} (1 - prod_j (1 - s_{g-1}(j))^{d_{k,o,j}})
```

A recovered clone is sampled top-down:

1. draw the founder type with weights `founder[k] · s_G(k)`;
2. at each cell, draw the outcome with weights `p_{k,o}(1 - prod_j(...))`;
3. draw the daughters' success flags conditional on at least one success;
4. recurse only into the daughters that succeed.

This draws exactly from the captured counts given detection, at a cost proportional to
the size of the observed tree. The founder mixture among recovered clones is therefore
reweighted by detectability and generally differs from `law.founder`.

## Bellman-Harris (`simulator=bellman_harris`)

This is an age-dependent process. A type-`k` cell born at time `t` lives
`L ~ Gamma(shape_k, mean_k / shape_k)` (mean `mean_k`, CV `1/sqrt(shape_k)`) and then
draws an outcome, and its daughters are born at `t + L`. The counts are of the cells
alive at the horizon `T`. The founder is born at `t = 0`.

* Physical parameters: the outcome probabilities plus `"<type>.mean_lifetime"`. The
  latter are inferred as `log mean ~ Uniform(log low, log high)`.
* `shape = 1`: Markov process with `E[Z_T] = founder · exp(T · diag(1/mean)(M - I))`
  (checked in the tests).
* `shape → ∞`: deterministic lifetimes, which reduce to Galton-Watson with generation
  time `mean` (checked in the tests).
* Clones under `selection: detected` are recovered by forward simulation plus rejection
  (`observation.max_attempts_per_clone`).

## Observation model

The order of operations is biology, then capture `Binomial(n_k, capture[k])`, then
labelling `Multinomial(·, confusion[k])`, then selection. An undetected endpoint cell is
not a death, and an undetected clone is not an extinct clone.

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
