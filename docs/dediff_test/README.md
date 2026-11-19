# Testing for dedifferentiation (B gives rise to A) with the exact GW likelihood

**Question for the wet lab.** Can KRT20⁺ cells (B) give rise to LGR5⁺ cells (A) at all, by any route?
Here: a likelihood-ratio test on organoid endpoint counts under the two-type Galton–Watson
model, its calibration, its power, and its sensitivity to cell-typing errors.
Study run 2026-11-18/19 (≈ 4.5 h, 16 workers); results in
`results/exact_gw/dediff_test/` (`summary/tables.md`, `summary/*.csv`); figures in
`figures/`. Code: `src/bio_sbi/inference/dediff_lrt.py`, scripts `scripts/dediff_test/`,
tests `tests/test_dediff_lrt.py` (11 pass).

## The test
- **H0 (no dedifferentiation):** B→A = B→AA = B→AB = 0 (B may die, persist, or divide into B+B).
- **H1:** at least one > 0. Effect size **m_BA = B→A + 2·B→AA + B→AB** — A cells produced by one
  B cell per generation.
- **Statistic:** Λ = 2[ℓ(full) − ℓ(H0)] ≥ 0, both maximised on the exact likelihood (adjoint
  kernel) with a multistart that includes θ̂₀, small steps along each route, pinned B→A starts and
  warm starts from the next smaller n. The full model's first B-row stick is the total
  dedifferentiation mass q, so H0 is exactly the face q = 0.
- **Calibration:** Monte-Carlo null (known H0 truth), parametric bootstrap from θ̂₀ (what one
  does with real data), and the boundary chi-bar-squared mixture with weights from the observed
  information at θ̂₀ (plus the textbook binomial(3,k)/8 weights and plain χ²₃).
- **Run on a dataset:** `uv run python scripts/dediff_test/run.py data counts.csv --reps 200
  [--confusion 0.02 --typing-kind b_as_a]` (CSV columns `horizon,A,B`, horizons in
  generations; prints Λ, m̂_BA, chi-bar / binomial / χ²₃ / bootstrap p-values).

## Design of the study
A row = reference (death .03, persist .03, to_B .05, A→AA .55, A→AB .05, A→BB .29). Null B rows:
N1 death .2 / persist .8; N2 death .1 / persist .5 / B→BB .4; N3 death .5 / persist .5;
N4 death .2 / persist .3 / B→BB .5. Alternatives on N1 and N2 with m_BA ∈ {0.0025, 0.005, 0.01,
0.02, 0.05, 0.1, 0.2, 0.4} by three routes (switch only, B→AB only, mixed). Generations
{4, 6, 8, 10}, n ∈ {128, 256, 512, 1024} clones per endpoint (nested datasets). 200 datasets
per null cell, 50 per alternative cell (10 for m ≥ 0.1), 100 per mis-typing cell.

## Results

### Calibration (false-positive rate at nominal 5%)
| null | 95% critical value, Monte Carlo (n = 128–1024) | chi-bar² (obs. information) | size, chi-bar² | size, binomial weights (5.43) | size, χ²₃ (7.81) | Λ = 0 |
|---|---|---|---|---|---|---|
| N1 | 1.9–3.4 | 3.3–3.5 | 1.0–5.5% | 0–3.5% | 0–2.5% | 65–77% |
| N2 | 3.0–4.1 | 3.7–3.9 | 3.5–6.0% | 1.0–3.0% | 0.5–1.0% | 48–59% |
| N3 | 4.1–4.8 | 3.8–4.2 | 5.0–6.5% | 1.0–3.5% | 0–1.5% | 25–36% |
| N4 | 2.3–3.6 | 3.6–3.9 | 2.5–5.5% | 1.5–2.5% | 0.5–1.5% | 44–52% |

- The **chi-bar² mixture with weights from the observed information** is close to nominal
  (1–6.5%). The textbook binomial weights and χ²₃ are conservative.
- The null distribution **depends on B's other outcomes**: using another null row's Monte-Carlo
  critical value gives up to 16% false positives. Calibrate per dataset (bootstrap or chi-bar²
  from the dataset's own fit), not with one fixed number.
- **Parametric bootstrap** (15 datasets × 100 replicates at n = 128 and 1024): median bootstrap
  critical values 1.5–4.0, close to the Monte-Carlo values except N1 at 1024 (2.3 vs 3.4) and
  N3 at 128 (3.2 vs 4.5); size with bootstrap critical values 5–11% — roughly calibrated, a little
  liberal at small n.
- Optimisation: in 3200 null fits no missed optimum was detected.

### Power
- **The test is extremely sensitive when typing is perfect.** Minimum detectable m_BA at 80%
  power: ≤ 0.0025–0.004 for every route at n = 128 (N1), ≤ 0.0025 for n ≥ 256 and for N2 at all
  n (`summary/tables.md`, `figures/power_mc.png`). At m_BA = 0.05, Λ is ≈ 80 at n = 128 and
  ≈ 670 at 1024.
- **Why:** the evidence comes from rare clones that are almost impossible without
  dedifferentiation (e.g. a clone with a single A cell and no B cell at generation 8 has
  log p ≈ −15 under H0); their score contributions are extremely heavy-tailed.

### Mis-typing (H0 true, LGR5/KRT20 classification errors in the data)
| analysis | false-positive rate at 5% (0.5–5% errors, n = 128–1024) |
|---|---|
| errors ignored | **75–100%** (already 75–100% at 0.5% "B read as A") |
| model assumes half the true error rate | **21–100%** |
| true confusion matrix in the model | 1–13% (≈ nominal from n = 256) |

The same rare clones that give the test its power are created by a single mis-typed cell
(one B read as A turns a common clone into an "impossible" one). **The detectable effects
(m_BA ≈ 0.003) are far below any realistic typing error rate, so the test is only as good as the
model of typing errors.**

### Real data (CRC0322, 2D, days 6/10 = generations 6/10, 165 + 101 organoids)
Λ = 45.1, bootstrap p = 0.005 (smallest possible with 200 replicates), m̂_BA = 1.35 with an odd
full-model fit (A mostly makes B+B; B→A 0.51, B→AA 0.42; one start reached it). **Not evidence**:
typing errors are not modelled, and the real organoids showed fate heterogeneity and density
dependence that the homogeneous model lacks.

## Recommendations for the wet lab
1. **Measure the LGR5/KRT20 confusion matrix** — above all the probability that a KRT20⁺ cell is
   scored LGR5⁺ — on the same assay (e.g. sorted pure populations, or double staining), with a
   precision well under a factor of 2 (modelling half the true rate still gives 21–100% false
   positives). Include it in the model.
2. **Clone numbers are not the bottleneck**: with correct typing, a few hundred clones per
   endpoint detect m_BA ≈ 0.005.
3. **A direct experiment is more robust**: organoids founded by sorted single KRT20⁺ cells (B
   founders), followed at early time points; any LGR5⁺ progeny with a validated marker is direct
   evidence, and the exact GW supports B founders for sizing it.
4. **Before applying the test to real data**, check robustness to organoid-level fate
   heterogeneity (it could create "impossible" clones like mis-typing does) — untested.

## Limitations
Homogeneous Galton–Watson model (discrete generations, no heterogeneity, no density dependence);
alternatives built on two null rows only; bootstrap on 15 datasets per cell; mis-typing levels
between "half" and "exact" not tested (how precisely the rate must be known is open).
