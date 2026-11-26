# Free founder distribution in the exact-likelihood GW posterior

**Question.** In organoid data the progenitor of a clone is hidden: only the endpoint
(A, B) counts are observed. The exact GW analysis assumed one A (LGR5⁺) founder per clone.
The misspecification study (`docs/misspecification`, report §10) showed that unmodelled
B founders or doublets push B→A along the weak direction (all-A NUTS on 70/30 founders:
T1 B→A = 0.02 ± 0.02 against 0.80) and that estimating the founder mixture by maximum
likelihood removes most of the bias. Here the founder distribution becomes a free
parameter of the **NUTS posterior**: P(founder = A) = p_A and the doublet fraction π₂,
with per-clone progenitor posteriors.

**Status (2026-11-23 12:00): complete, reduced.** n = 512 clones per endpoint, 9 XG2
truths, dataset 0 (41 NUTS runs). Main runs complete: base / F70 with free p_A, D15A /
D15mix with free p_A + π₂. Controls reduced by the deadline: F70 with π₂ free on T1_ref and
T2_toA05 only (T3_Bdiv, T4_slowA, P3 were started and killed at 12:00); base with π₂ free
dropped; dataset 1 only for F70 / free p_A on T1_ref, T2_toA05, T4_slowA. No other
experiment was rerun. All runs were stopped at 11:51, when the main local list finished
(the 12:00 cutoff left no time for the remaining controls).

## Results

**Founder parameters** (`figures/founder_posteriors.png`, `tables/founder_posteriors.csv`;
mean [90% interval], truth p_A = 0.7 for F70 / D15mix, 1 for base / D15A, π₂ = 0.15):

| scenario, model | p_A: typical posterior | p_A covered (90%) | π₂ | π₂ covered |
|---|---|---|---|---|
| base, free p_A | 0.34–0.93; T2/T4/T5 0.89–0.93 [≈0.75, 0.99], others ≈ [0.02–0.1, 0.92–0.99] | 0/9 (truth on the boundary) | — | — |
| F70, free p_A | 0.35–0.76; e.g. T1 0.35 [0.08, 0.58], T5 0.60 [0.46, 0.75] | 7/9 (not T1, T4) | — | — |
| F70, free p_A + π₂ (T1, T2) | 0.34 [0.07, 0.57], 0.64 [0.44, 0.81] | 1/2 | **0.005 [0.000, 0.014], 0.021 [0.001, 0.059]** (truth 0) | — |
| D15A, free p_A + π₂ | 0.46–0.87, intervals reaching 0.91–0.99 | 0/9 (boundary) | 0.13–0.20 (T2: 0.37 [0.16, 0.88]) | 7/9 |
| D15mix, free p_A + π₂ | 0.22–0.73; T1 0.22 [0.02, 0.51], T4 0.24 [0.02, 0.58] | 6/9 | 0.14–0.42 | 5/9 |

- **p_A is weakly identified, as in the MLE study**: on the P truths (low B→A) the
  posterior is almost the prior (90% intervals ≈ [0.05, 0.95]); it is informative only
  for T2_toA05 and T5_C7 (intervals 0.3–0.35 wide). Where B→A is large (T1, T4) the
  posterior prefers p_A ≈ 0.2–0.4 and misses 0.7 (the study's extended MLE found 0.21–0.51 on the
  same datasets). A dataset-1 replicate of F70 gives the same picture (T1 0.28
  [0.05, 0.52], T4 0.30 [0.03, 0.60], T2 0.53 [0.34, 0.72]).
- **The doublet fraction is identified**: π₂ intervals are 0.05–0.25 wide around 0.15 for
  most truths (doublet clones are visibly larger), and with π₂ free on 70/30 single-founder
  data **the model does not invent doublets** (π₂ ≤ 0.06 at the 95th percentile; only 2
  truths, see status).
- **Branches.** Replaying the 8 L-BFGS starts (`optima.json`) shows two branches along
  the "parent swap" direction: (a) p_A ≈ 0.2–0.6 with B→A near the truth, and (b) p_A ≈
  0.8–0.95 with B→A ≈ 0.02 (B founders replace dedifferentiation; the all-A model's
  answer). Where both were found, (b) lies 3–12 log-likelihood units below (a) (F70 T1 11.6,
  T3 4.7; D15mix T1–T4 2.8–9.2); the chains started on (a), except single chains of
  F70 T3_Bdiv, base T3_Bdiv and D15A T4_slowA (see mixing). For the all-A-founder data
  the ridge is flatter: base T1's best optimum is p_A = 0.30, B→A = 0.08; D15A T4's is
  p_A = 0.19, B→A = 0.03. The log-likelihood at the truth is within 7 units of the best
  L-BFGS optimum in every run (typical for 11–12 parameters): the truth lies on the flat
  ridge.

**B→A, m_BA, growth rate ρ, stable A fraction** (posterior, dataset 0, 9 truths;
`figures/b_to_a.png`, `tables/key_summary.csv`):

| | B→A mean bias / max abs / max abs z / 90% cover | m_BA cover | ρ cover | stable A cover |
|---|---|---|---|---|
| F70, **all-A NUTS** (study) | −0.24 / 0.78 / **38.6** / 5/9 | 6/9 | 5/9 (max abs z 3.0) | 8/9 |
| F70, free p_A | −0.09 / 0.28 / 2.8 / 5/9 | 9/9 | 7/9 | 8/9 |
| D15A, free p_A + π₂ | −0.17 / 0.61 / 3.5 / 4/9 | 9/9 | 8/9 | 9/9 |
| D15mix, free p_A + π₂ | −0.08 / 0.26 / 2.4 / 7/9 | 9/9 | 9/9 | 8/9 |
| base, free p_A (correctly specified data) | −0.17 / 0.54 / 3.2 / 6/9 | 6/9 | 9/9 | 9/9 |
| MLE, all-A model (study, datasets 0–9; mean / max abs bias) | F70 −0.18 / 0.80, D15A −0.28 / 0.80, D15mix −0.30 / 0.80, base −0.04 / 0.80 | | | |
| MLE, extended model (study, datasets 0–9) | F70 −0.04 / 0.46, D15A −0.04 / 0.80, D15mix −0.05 / 0.60 | | | |

- **The confidently wrong all-A posterior is gone.** F70 T1: all-A NUTS B→A 0.02
  [0.00, 0.06] (z = −39) → free p_A 0.52 [0.36, 0.69] (z = −2.8); T3 0.06 → 0.38, T4
  0.04 → 0.57. D15mix T1 0.54 [0.37, 0.72], T4 0.64 [0.46, 0.78] (all-A MLE: 0.00).
  The worst |z| drops from 39 to ≤ 3.5.
- **But B→A stays biased low where it is large** (T1–T4: posterior means 0.15–0.35
  below the truth, coverage 0/4 in F70): the flat p_A prior puts mass on the
  low-p_A / lower-B→A part of the ridge. The P truths (B→A ≈ 0.1–0.3) are covered
  everywhere. m_BA (always covered in the founder scenarios), ρ and the stable A fraction
  are robust (|bias| ≤ 0.02 for ρ, ≤ 0.005 for stable A).
- **Cost on correctly specified data.** On base (all founders A), freeing p_A biases
  B→A for T1 (0.26 [0.03, 0.62] vs 0.80), T2 (0.27 vs 0.50) and T3 (0.14 vs 0.60):
  along the ridge the likelihood is nearly flat (base T1: best optimum p_A = 0.30,
  B→A = 0.08, only 0.5 units above the truth p_A = 1, B→A = 0.8), so the uniform prior on
  p_A pulls B→A down. The all-A model is correct there (MLE mean bias −0.04). **Free p_A
  trades a confident bias (when founders are mixed) for a wide, downward-shifted
  posterior (when they are not).** An informative p_A prior (founder identity recorded at
  seeding, or sorting purity) is what makes B→A identifiable.

**Per-clone progenitor posterior** (averaged over 100 posterior draws; AUC / Brier against
the simulated true founders, mean over truths; "constant" = the same mean probability for
every clone; oracle = at the true parameters; `figures/progenitor_*.png`,
`tables/progenitor*.csv`):

| task | scenario | AUC | Brier | Brier constant | oracle AUC / Brier |
|---|---|---|---|---|---|
| all founders A (vs any B) | F70, free p_A | 0.61 | 0.232 | 0.241 | 0.61 / 0.194 |
| | F70, free p_A + π₂ (T1, T2 only; free p_A on the same two: 0.72) | 0.72 | 0.238 | 0.257 | 0.72 / 0.167 |
| | D15mix | 0.59 | 0.304 | 0.308 | 0.59 / 0.213 |
| doublet | D15A | 0.78 | 0.109 | 0.145 | 0.78 / 0.103 |
| | D15mix | 0.78 | 0.127 | 0.159 | 0.78 / 0.107 |

- **Founder type is barely recoverable per clone** (AUC ≈ 0.6 even at the true
  parameters): a B-founded clone that switches to A early looks like an A clone that is
  about one generation behind. This is why p_A is weakly identified. By truth: AUC
  0.68–0.75 (F70) / 0.61–0.65 (D15mix) for T1–T4, ≈ 0.52 (no information) for the four P
  truths. The ranking of the fitted posterior
  matches the oracle (same AUC), but its **calibration is off** where p_A is
  underestimated (D15mix: mean P(all A) 0.42 vs true rate 0.67; reliability curves lie
  above the diagonal); Brier gains over a constant are small (≤ 0.02).
- **Doublets are recoverable**: AUC 0.78 and Brier 0.11–0.13 vs 0.15–0.16 for a constant,
  well calibrated below 0.5, slightly over-confident at 0.6–0.9.
- **Generation**: early endpoints are slightly more informative for founder type (F70 AUC
  0.64 at generation 4 vs 0.60 at 10; D15mix 0.61 vs 0.57; same in the oracle), not for
  doublets (0.78–0.80 at every generation; by truth 0.66–0.87). See
  `figures/progenitor_auc.png`.

**Mixing** (`tables/mixing.csv`; poorly mixed = rank R-hat > 1.05, bulk ESS < 100 or chain
means > 0.5 sd apart): 20 of 41 runs are flagged. D15A 8/9 (max R-hat 1.76, T4_slowA
has one chain at p_A 0.19 and three at 0.91, i.e. on the other branch), D15mix 4/9
(T5_C7 R-hat 1.54), F70 3/12 (T3_Bdiv R-hat 1.30, chains at p_A 0.24–0.56),
F70 + π₂ 1/2 (T2 R-hat 1.58), base 4/9 (T3_Bdiv chains at p_A 0.15–0.68). Only 5 runs met
the stopping targets within 500 draws per chain; most stopped at max_draws with R-hat
1.01–1.05. Divergences 0–242 per run (mostly < 50); tree depth 7 often hit (the ridge).
Posterior summaries from poorly mixed runs (especially D15A) under-represent one branch
and should be read as indicative. Run time per job (4 threads): median 19 min (F70) to
53 min (base, slower remote host), max 102 min; one free-founder gradient costs ≈ 1.4× the
fixed-founder one.

**What was skipped.** Datasets 1–2 (except 3 F70 replicates), base with π₂ free, F70 with
π₂ free on 7 of 9 truths, the brow / probit reparametrisations with free founders, a p_A
prior other than uniform, longer chains for the poorly mixed runs.


## Implementation

**Model.** With founder types φ = (p_A, 1 − p_A) and founder sizes m ∈ {1, 2} with
probabilities (π₁, π₂), the grid values of the clone pgf are

  G(x, y) = Σ_m π_m S^m,  S = p_A F_A(x, y) + (1 − p_A) F_B(x, y),

with F_k the pgf of a clone started by one type-k cell (already computed by the exact
likelihood, `gw_exact.py`). The detection probability is mixed the same way,
q = Σ_m π_m (1 − (1 − φ·g)^m), g_k = P(a type-k cell leaves a detected descendant).

**Gradients** (exact, no finite differences):
∂G/∂φ_k = (Σ_m π_m m S^{m−1}) F_k, ∂G/∂π_m = S^m, and
∂q/∂φ_k = (Σ_m π_m m (1 − s)^{m−1}) g_k, ∂q/∂π_m = 1 − (1 − s)^m (s = φ·g), chained
through the stick-breaking coordinates of the founder rows.
`gw_exact.log_likelihood_founder` implements both kernels:
- `adjoint` (used for all runs): F_A and F_B come from the stored final pgf iterate of a
  single-founder evaluation; the founder gradients are the FFT adjoint (the same cotangent
  as the outcome-row gradient) contracted with ∂G/∂φ and ∂G/∂π — final-generation
  quantities only. The outcome-row gradient uses the existing reverse sweep. Cost per
  gradient evaluation ≈ 1.4–2× the fixed-founder likelihood.
- `loop`: F_A and F_B are propagated separately in forward mode (2× cost).
Detection: g_A, g_B (and their outcome-row derivatives) from the existing complement-form
detection recursion with a single founder of each type.

**Opt-in, default unchanged.** `inference=exact_gw` used to refuse free founder rows; it now
accepts them (`'~law.fixed.founder'` frees founder.A; `law.founder_sizes=[1,2]` adds the
founder-size row). With fixed founder rows (every existing config) the likelihood takes
the original code path: same function, same values and draws (tested bit for bit).
Free founder rows need the default NUTS coordinates (`reparam: none`; the brow/probit
reparametrisations and the Fisher-information code raise). Priors: the founder rows are
ordinary Dirichlet rows of the parameter space, so `law.concentration` sets them.
**Choice here: Uniform(0, 1) for p_A and for π₂** (concentration 1). A Beta(1, 4) on π₂
would encode "doublets are rare"; the flat prior lets the data decide and is the honest
test of whether the model invents doublets (base / F70 with π₂ free).

**Per-clone progenitor posterior.** `ExactLikelihood.progenitor_posterior(theta)` returns,
for every observed (A, B) cell of every generation, P(founders = c | counts) over the
compositions c ∈ {A, B} (sizes {1}) or {A, B, AA, AB, BB} (sizes {1, 2}):
weights π_m C(m, k) p_A^k (1 − p_A)^{m−k} times the law of F_A^k F_B^{m−k} at the cell
(`gw_exact.founder_composition_laws`, one extra FFT per composition; detection
conditioning cancels). For size-1 founders this is p_A L_A / (p_A L_A + (1 − p_A) L_B).

**Tests** (`tests/test_free_founder.py`, fast): gradients against central finite
differences for both kernels and both founder-size sets (all 11/12 coordinates, rel 1e-4);
free model = fixed-founder model at the same founder distribution (value and outcome
gradient, both kernels); fixed-founder path bit-for-bit the original kernel; reparam
raises; progenitor posterior against brute-force enumeration (one generation, sizes
{1, 2}) and against the closed form for single founders. `ruff check`, `ruff format
--check` and `pytest -m "not slow"` (367 passed) pass.

## Runs

**Data.** The founder scenarios of the misspecification study, same simulator and seeds
(the counts are replayed and checked to be identical to the study's datasets; the true
founder count and types of every clone are recorded): n = 512 detected clones per
generation 4/6/8/10, the 9 XG2 truths, dataset 0.

| scenario | truth | models |
|---|---|---|
| base | one A founder | fA (p_A free), fA+d2 (p_A, π₂ free) |
| F70 | one founder, A/B 70/30 | fA, fA+d2 |
| D15A | 15% doublets, A founders | fA+d2 |
| D15mix | 15% doublets, every founder A/B 70/30 | fA+d2 |

Run as planned: base and F70 with free p_A, D15A and D15mix with free p_A + π₂ (all 9 truths); controls reduced (see status).

**NUTS.** As the study's all-A runs where possible: adjoint kernel, 4 chains started from
the best of 8 L-BFGS optima of prior draws (Laplace draws around them), 300 warmup
iterations, tree depth ≤ 7, flat priors; plus the ESS/R-hat stopping rule
(`inference.method.stop`: rank R-hat ≤ 1.01, bulk ESS ≥ 200, tail ESS ≥ 100, chain-mean
spread ≤ 0.5 posterior sd, checked every 50 draws, at most 500 draws per chain). The
study's all-A runs drew a fixed 500 per chain (its script did not pass the stop rule to
`run()`). Branches: `optima.py` replays the 8 L-BFGS starts and records where each optimum
lies (p_A, π₂, B→A) and the log-likelihood at the truth.

**Reproduce.**

    uv run python scripts/free_founder/run.py --jobs D15mix:fA+d2,F70:fA --seeds 0
    uv run python scripts/free_founder/optima.py
    uv run python scripts/free_founder/analyse.py

Results: `/home/matteo/BIO-SBI/results/exact_gw/free_founder/<scen>/<model>/<gt>_s<seed>.{json,npz}`
(json: posterior summaries, correlations, convergence, run time; npz: draws, chain index,
per-clone generation / counts / true founder count / true A founders, progenitor
posterior averaged over 100 draws and at the posterior mean), `optima.json`, `logs/`.
Tables: `tables/`, figures: `figures/`.
