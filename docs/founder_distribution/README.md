# Founder (seeding) distribution: how it affects inference in the exact two-type GW model

**Question.** In the organoid experiment a clone is seeded by m = 1…5 cells (size
distribution π_m), each founder cell is A (LGR5⁺) with probability p_A, B (KRT20⁺)
otherwise, and only the endpoint (A, B) counts are observed. How does the founder
distribution change what the counts tell us about the transitions? Is a non-trivial founder
distribution more or less informative than single A founders (the XG2 setting), how much
does it cost to *estimate* it (the realistic case: nothing about the seeding is recorded,
only its probabilities can be learnt from the counts), which founder parameters are
identifiable, and do several seeding conditions with different, unknown founder
distributions help?

**Status (2026-11-25):** complete; simulation check reduced. Fisher study: all 9 truths × 32 founder
distributions (4 size distributions × p_A ∈ {1, .9, .7, .6, .5, .3, .1, 0}) × 2 designs ×
known / estimated, validated to 3e-13 against the single-founder information and to 1e-8
against finite differences of the exact log-likelihood. Multiple conditions: Fisher (9 truths,
6 combinations) and joint MLE (2 truths, 12 datasets). Simulation check: 16 NUTS runs
(dataset 0; reduced from the planned 24, see Section 3) and 288 MLE fits (3 truths × 4
distributions × est / known × 12 datasets). Local machine only.

## Summary for the lab (plain language)

1. **Seed single cells.** Seeding small groups is always worse, even if the group sizes were
   known exactly: a clone grown from m cells is the sum of m clones, and only the total is
   seen. With the current endpoints (generations 4/6/8/10), "mostly single" seeding (70%
   single, 20% pairs, …) with an unknown size distribution needs about 2–3× more clones for
   the same precision along the weakest direction (B→A, the B row), uniform 1–3 cells about
   4×, uniform 1–5 cells about 7×. In simulations with 512 clones per time point, the B→A
   error for the dedifferentiation truths is 2–3× larger when 1–5 cells are seeded and the
   seeding distribution must be estimated than when it is known.
2. **If founders are not recorded, seeding single cells matters more than sort purity.**
   Single cells with an unknown A/B proportion lose 12–27% of the information (Fisher). The
   proportion itself is learnt only for truths with strong dedifferentiation (±0.1), and not
   at all otherwise. If the proportion is *known* (for example, sort purity measured
   independently), B founders even add information on B→A at the late endpoints (up to 3×
   smaller variance with only B founders). So record or measure purity whenever possible;
   a recorded founder type per clone is best.
3. **Early time points identify the founders.** Counting some clones after 1–2 divisions
   (the R1 design) pins down the A/B proportion (±0.03 for the dedifferentiation truths) and
   the group-size distribution. It is also far more informative overall (7–10× the XG2
   design). If the founders cannot be recorded, add early time points.
4. **Several seeding conditions (e.g. A-enriched and A-depleted sorts) help when the
   alternative is one mixed population.** They give 1.2× (late endpoints) to 1.75× (with
   early time points) more information than one condition at the average proportion; in
   simulations the B→A error fell by about 40%. They do not beat a single well-sorted,
   A-enriched condition with the same number of clones, and varying the seeding *size*
   across conditions does not help. Several conditions are worth it if a clean single-type
   sort is impossible; otherwise put the clones into the enriched, single-cell condition.

## Method

### Model

G(x, y) = Σ_m π_m S^m, S = p_A F_A + (1 − p_A) F_B, with F_k the pgf of a clone seeded by
one type-k cell (the exact-GW pgf of `gw_exact.py`): founder cells' types are i.i.d., and an
m-founder clone is the m-fold convolution of single-founder clones, i.e. S^m on the FFT
grid. Detection: q = Σ_m π_m (1 − (1 − s)^m), s = p_A g_A + (1 − p_A) g_B. This is exactly
the likelihood of the free-founder NUTS (`gw_exact.log_likelihood_founder`).

### Fisher information with a founder distribution (extension of `optimal_design/founder_uncertainty.py`)

`scripts/founder_distribution/fisher.py`. For one detected clone observed at generation h:

  I = Σ_c p̃_c s_c s_cᵀ,  p̃_c = p_c / q,  s_c = ∂p_c / p_c − ∂q / q,

in 15 coordinates: the 10 transition directions of the optimal-design study (orthonormal
row contrasts U, prior whitening by 42), p_A, and 4 orthonormal contrasts of the
founder-size simplex. The derivatives are computed on the pgf grid and inverted by FFT:

- ∂G/∂θ = G′(S) (p_A ∂F_A + (1 − p_A) ∂F_B), G′(S) = Σ_m π_m m S^{m−1}
  (∂F_k in forward mode, `_loop_single_founder`);
- ∂G/∂p_A = G′(S) (F_A − F_B), ∂G/∂π_m = S^m;
- detection mixed the same way (∂q/∂p_A = Σ_m π_m m (1 − s)^{m−1} (g_A − g_B),
  ∂q/∂π_m = 1 − (1 − s)^m).

F_A, F_B and their 10 derivatives are computed once per (truth, generation) on a square
undamped grid large enough for 5 founders (P(N ≥ L | detected) ≤ 1e-12 for 5 A or 5 B
founders; L up to 2352 for T5 at generation 10); every founder distribution is then a
pointwise polynomial plus 16 inverse FFTs. The whole study (9 truths × 6 generations × 32
founder distributions) takes 2 min on 3 cores.

- **Known (K)**: information on the transitions = the 10 × 10 block.
- **Estimated (E)**: p_A and the size probabilities on the *true support* (S1: none;
  S3: sizes 1–3; S2, S4: 1–5) are nuisances, profiled by the Schur complement. Support
  restriction: a size probability that is truly 0 is a boundary parameter, and at early
  generations clones larger than m·2^h are impossible without m-founder clones, so the
  information on π_4, π_5 at S3 is non-regular (the validation shows the expected score is
  then not zero). The NUTS check estimates the sizes on the same support. p_A = 1 and p_A = 0
  are boundary values too; their E numbers use the two-sided (regular) information and are
  approximate. E_type (only p_A estimated) and E_size (only sizes) decompose the cost.
- **Truths**: the 9 XG2 truths; for T1, T2, T4 (B divisions exactly 0) the eps = 0.01
  variant as in the optimal-design study (B-founded information is non-regular at the face).
- **Designs**: XG2 = A-founded XG2 endpoints 4/6/8/10, 1/4 each; R1t = the time points of
  the optimal design R1 (generations 1, 2, 10 with shares 1/4, 5/8, 1/8), founders not
  labelled. N = 2048 clones in both (512 per XG2 endpoint).
- **Reported**: λ_min of the prior-whitened transition information (1/λ_min = clones per
  unit of prior information along the weakest direction, report §6.3); asymptotic sd of
  B→A, m_BA (mean A daughters of a B cell), growth rate ρ, stable A fraction at N = 2048;
  for E the sd of p_A and of the mean founder size Σ m π_m (joint with the transitions, and
  with the transitions known). Gains are relative to **single A founder, known** in the same
  design ("same design") or in XG2.

**Validation** (`validate_fisher.py`, `tables/fisher_validation.json`):

| check | result |
|---|---|
| S1, p_A = 1 (single A founder) vs `od_common.information_basis` (P3 h 4/8, T1 h 6, T3 h 10) | rel. error ≤ 3.0e-13, detection identical |
| S1, p_A = 0.7 vs `founder_uncertainty.mix_info` (transitions + p_A, 11 × 11) | rel. error ≤ 3.0e-13 |
| score of a sampled histogram (sizes 1–5: S4/0.5 P3 h4, S2/0.7 T1 h6, S3/0.9 T3 h4, S4/0.3 T5 h2) vs the exact log-likelihood gradient, all 15 coordinates | 2e-15 (`loop` kernel), 7e-14 (`adjoint`) |
| same vs central finite differences of the log-likelihood value | ≤ 1.1e-8 |
| kept conditional mass | ≥ 1 − 2.2e-10 |
| expected score | ≤ 3e-9 (interior); 0.04 at S3 with π_4, π_5 = 0 (non-regular boundary, see above) |

## Results

### 1. Information vs founder distribution (Fisher, 9 truths, 2 designs)

Figures: `figures/info_lam.png` (λ_min), `figures/info_btoa.png` (1/Var B→A),
`figures/info_mba.png`, `figures/info_rho.png` — heatmaps over size distribution × p_A,
known vs estimated, median over truths of the gain relative to single A founders, known
(same design); `figures/cost_per_truth.png` (E/K per truth). Tables: `tables/fisher_key.md`
(below), `tables/fisher_summary.csv` (all 32 distributions, 4 modes, both designs, median /
min / max over truths), `tables/fisher_main.csv` (every truth), `tables/fisher_per_truth_*.md`.

Median over the 9 truths; "gain" = ratio to single A founder known in the same design
(λ_min ratio; inverse-variance ratio for B→A, m_BA, ρ); E/K = cost of estimating, range over
truths; sd of the founder parameters at N = 2048 for estimated founders, median over the T
(T1–T5) and P (P1–P4) truths:

| design   | founders    |   λ gain K |   Var(B→A) gain K |   λ gain E |   Var(B→A) gain E |   Var(m_BA) gain E |   Var(ρ) gain E | E/K λ (range over truths)   | sd p_A: T / P truths   | sd mean size: T / P   |
|:---------|:------------|-----------:|------------------:|-----------:|------------------:|-------------------:|----------------:|:----------------------------|:-----------------------|:----------------------|
| XG2      | S1, p_A 1.0 |       1.00 |              1.00 |       0.86 |              0.94 |               0.82 |            0.92 | 0.68–1.00                   | 0.09 / 1.15            | –                     |
| XG2      | S1, p_A 0.9 |       1.02 |              0.99 |       0.79 |              0.78 |               0.79 |            0.91 | 0.50–0.98                   | 0.10 / 1.10            | –                     |
| XG2      | S1, p_A 0.7 |       1.05 |              1.03 |       0.76 |              0.77 |               0.77 |            0.90 | 0.32–1.00                   | 0.12 / 0.86            | –                     |
| XG2      | S1, p_A 0.5 |       1.04 |              1.35 |       0.73 |              0.72 |               0.82 |            0.90 | 0.33–1.00                   | 0.14 / 0.91            | –                     |
| XG2      | S1, p_A 0.3 |       1.09 |              1.68 |       0.83 |              0.80 |               0.79 |            0.89 | 0.36–1.00                   | 0.11 / 1.17            | –                     |
| XG2      | S1, p_A 0.0 |       1.07 |              3.00 |       0.88 |              1.27 |               1.01 |            0.86 | 0.38–0.99                   | 0.08 / 1.72            | –                     |
| XG2      | S2, p_A 1.0 |       0.79 |              0.72 |       0.46 |              0.55 |               0.34 |            0.53 | 0.27–0.84                   | 0.12 / 1.69            | 0.06 / 0.23           |
| XG2      | S2, p_A 0.9 |       0.80 |              0.67 |       0.42 |              0.47 |               0.34 |            0.53 | 0.36–0.72                   | 0.13 / 1.54            | 0.06 / 0.21           |
| XG2      | S2, p_A 0.7 |       0.67 |              0.68 |       0.35 |              0.40 |               0.34 |            0.52 | 0.31–0.87                   | 0.18 / 1.34            | 0.07 / 0.14           |
| XG2      | S2, p_A 0.5 |       0.67 |              0.76 |       0.41 |              0.48 |               0.37 |            0.54 | 0.31–0.99                   | 0.24 / 1.43            | 0.12 / 0.11           |
| XG2      | S3, p_A 0.9 |       0.52 |              0.41 |       0.24 |              0.32 |               0.24 |            0.77 | 0.23–0.70                   | 0.17 / 2.22            | 0.15 / 0.47           |
| XG2      | S3, p_A 0.5 |       0.45 |              0.43 |       0.26 |              0.26 |               0.25 |            0.75 | 0.36–0.99                   | 0.25 / 2.02            | 0.16 / 0.22           |
| XG2      | S4, p_A 0.9 |       0.34 |              0.28 |       0.14 |              0.18 |               0.12 |            0.78 | 0.17–0.62                   | 0.22 / 3.17            | 0.37 / 1.22           |
| XG2      | S4, p_A 0.5 |       0.29 |              0.26 |       0.15 |              0.15 |               0.13 |            0.76 | 0.25–0.99                   | 0.38 / 2.53            | 0.49 / 0.63           |
| R1t      | S1, p_A 1.0 |       1.00 |              1.00 |       0.94 |              0.84 |               0.77 |            0.98 | 0.41–0.99                   | 0.02 / 0.16            | –                     |
| R1t      | S1, p_A 0.9 |       0.89 |              0.86 |       0.74 |              0.73 |               0.67 |            0.88 | 0.60–1.00                   | 0.03 / 0.13            | –                     |
| R1t      | S1, p_A 0.7 |       0.82 |              0.70 |       0.53 |              0.46 |               0.49 |            0.84 | 0.38–1.00                   | 0.03 / 0.09            | –                     |
| R1t      | S1, p_A 0.5 |       0.66 |              0.83 |       0.30 |              0.37 |               0.33 |            0.84 | 0.24–1.00                   | 0.04 / 0.09            | –                     |
| R1t      | S1, p_A 0.3 |       0.62 |              0.96 |       0.23 |              0.30 |               0.28 |            0.59 | 0.37–1.00                   | 0.05 / 0.14            | –                     |
| R1t      | S1, p_A 0.0 |       0.29 |              0.58 |       0.28 |              0.37 |               0.36 |            0.78 | 0.18–0.99                   | 0.16 / 0.26            | –                     |
| R1t      | S2, p_A 1.0 |       0.84 |              0.76 |       0.68 |              0.64 |               0.60 |            0.66 | 0.09–0.94                   | 0.02 / 0.16            | 0.02 / 0.05           |
| R1t      | S2, p_A 0.9 |       0.94 |              0.64 |       0.52 |              0.52 |               0.49 |            0.60 | 0.18–0.94                   | 0.03 / 0.14            | 0.02 / 0.04           |
| R1t      | S2, p_A 0.7 |       0.54 |              0.51 |       0.39 |              0.26 |               0.31 |            0.57 | 0.38–0.92                   | 0.04 / 0.09            | 0.02 / 0.03           |
| R1t      | S2, p_A 0.5 |       0.47 |              0.40 |       0.22 |              0.24 |               0.19 |            0.51 | 0.22–0.99                   | 0.05 / 0.08            | 0.03 / 0.03           |
| R1t      | S3, p_A 0.9 |       0.80 |              0.48 |       0.29 |              0.33 |               0.35 |            0.90 | 0.05–0.85                   | 0.03 / 0.15            | 0.04 / 0.13           |
| R1t      | S3, p_A 0.5 |       0.30 |              0.31 |       0.13 |              0.12 |               0.13 |            0.83 | 0.08–0.99                   | 0.06 / 0.08            | 0.06 / 0.04           |
| R1t      | S4, p_A 0.9 |       0.51 |              0.31 |       0.18 |              0.21 |               0.26 |            1.00 | 0.08–0.89                   | 0.04 / 0.15            | 0.09 / 0.27           |
| R1t      | S4, p_A 0.5 |       0.21 |              0.24 |       0.08 |              0.09 |               0.10 |            0.93 | 0.08–0.95                   | 0.07 / 0.08            | 0.13 / 0.08           |

Reference (single A founder known, N = 2048): XG2 1/λ_min 384–11983 clones (T truths
384–1536, P truths 1464–11983), sd(B→A) 0.03–0.16; R1 time points 1/λ_min 76–6768,
sd(B→A) 0.02–0.03 — the R1 time points are 7× (λ_min) / 10× (Var B→A) more informative
than the XG2 endpoints even with every clone A-founded.

**Is a known non-trivial founder distribution more or less informative than single A founders?**

- **Type proportion, single cells, known p_A: depends on the design.** With the late XG2
  endpoints, B founders *add* information on B's row: λ_min ×1.02–1.09 and Var(B→A) ÷1.35
  (p_A 0.5) to ÷3.0 (p_A 0, all B). The gain is concentrated on the T truths (large B→A;
  p_A 0: λ_min ×3.7, Var(B→A) ÷4.5) and absent for the P truths (×0.97). With the early R1
  time points B founders cost information (λ_min ×0.89 at p_A 0.9, ×0.66 at 0.5, ×0.29 at 0):
  a B-founded clone observed after 1–2 generations has few A cells, so the A row is learnt
  from A-founded clones; at late time points the A row is learnt anyway, and a B-founded
  clone directly shows how B cells switch.
- **Multi-cell seeding, known sizes: always less informative.** Mostly single (S2)
  λ_min ×0.79 (XG2) / ×0.84 (R1), uniform 1–3 ×0.52 / ×0.80, uniform 1–5 ×0.34 / ×0.51 at
  p_A ≈ 0.9–1 (Var(B→A) ×0.72 → ×0.28 on XG2): a clone of m founders is the sum of m
  independent clones and only their sum is observed, so a multi-founder clone carries less
  information than one single-founder clone although it has more cells. Exception: R1 time
  points with only B founders (S3/S4 at p_A = 0, ×1.4–1.6 relative to single A founders;
  more cells per clone partly compensate for the uninformative B-only early clones; a corner
  case).
- The growth rate ρ is robust: Var(ρ) stays within ×0.5–2.2 of the reference in every case
  (known or estimated).

**How much does estimating the founder distribution cost?**

- **Single cells, p_A unknown (S1, E):** λ_min ×0.73–0.88 of the single-A-known reference on
  XG2 (median over truths; per truth E/K 0.32–1.00), Var(B→A) ×0.72–0.94 (×1.27 at p_A 0:
  even estimated, B founders help the T truths). On the R1 time points the cost grows with
  the B share: ×0.94 (p_A 1) → ×0.53 (0.7) → ×0.30 (0.5).
- **Mostly single (S2), everything unknown:** λ_min ×0.35–0.46 (XG2) / ×0.22–0.68 (R1): about
  half of the information is lost; on XG2, E/K is 0.27–0.99 per truth. The cost splits
  roughly evenly between not knowing p_A (E_type alone: λ ×0.49–0.74) and not knowing the
  sizes (E_size alone: ×0.49–0.65; `tables/fisher_summary.csv`).
- **Uniform 1–3 / 1–5 (S3/S4), unknown:** λ_min ×0.24–0.26 / ×0.14–0.15 on XG2, i.e. 4–7× more
  clones for the same precision along the weakest direction; Var(m_BA) ×0.12–0.25.
- **What gets lost.** B→A and m_BA (the B row, the weak directions) carry the cost; ρ and the
  stable A fraction are much less affected (Var(ρ) ×0.5–1.0). The cost is largest where the
  B row is identified mainly through B→A (T1–T4: per-truth Var(B→A) E/K down to 0.27, T1/T2/T3
  at S4/0.5), smallest for T5 and P1 (Var(B→A) E/K ≈ 1).

**Which founder parameters are identifiable** (`figures/founder_identifiability.png`)?

- **XG2 endpoints: p_A is identified only for the T truths** (sd 0.08–0.14 for single cells,
  0.12–0.24 for S2, 0.22–0.38 for S4) and **not at all for the P truths** (sd 0.9–3.2, far
  wider than the [0, 1] prior: the posterior is the prior, as the free-founder NUTS study
  found). Knowing the transitions halves sd(p_A) for single cells (0.07), so p_A is
  confounded with the transitions (the parent-swap direction: B founders vs B→A); for
  multi-cell seeding it stays poorly identified even with the transitions known (S4: 0.4–0.6).
- **The mean founder size is identified** on XG2 for S2 (sd 0.06–0.12 T, 0.11–0.23 P around
  1.44) and S3 (0.15 T), poorly for S4 (0.4–1.2 around 3): multi-founder clones are visibly
  larger, as π₂ in the free-founder study.
- **Early time points identify both**: on the R1 time points sd(p_A) is 0.02–0.05 for the T
  truths (p_A ≥ 0.3) and 0.08–0.16 for the P truths, sd(mean size) 0.02–0.27 — clones
  observed after 1–2 generations are small and still close to their founders (count and
  types).

**Dependence on the truth and the design.** The P truths (small B→A, B rarely switches) are
little affected by B founders on XG2 (the B row is learnt from A-derived B cells) but their
p_A is unidentifiable there; the T truths (large B→A) gain most from known B founders and
lose most when p_A must be estimated. Early time points (R1) make the founder distribution
learnable (sd p_A ≈ 0.03) and keep the cost of estimating it moderate for p_A ≥ 0.9, but are
hurt by a large B share (the early A-founded clones carry the A row).

### 2. Several seeding conditions (Fisher)

Shared transitions, each condition with its own unknown p_A and size probabilities
(Schur complement per condition, summed), N = 2048 clones split equally; compared with one
condition of the same N (`figures/multi_conditions.png`, `tables/multi_conditions_ratios.md`,
`tables/multi_conditions.csv`). Ratios are medians over truths (combined / single, both
estimated; >1 = combined better):

| design   | conditions (estimated)                              |   E/K λ | vs single 1   |   λ ratio 1 |   Var(B→A) ratio 1 | vs single 2   |   λ ratio 2 |   Var(B→A) ratio 2 | vs single 3   |   λ ratio 3 |   Var(B→A) ratio 3 | sd p_A per condition (T)   | sd p_A per condition (P)   |
|:---------|:----------------------------------------------------|--------:|:--------------|------------:|-------------------:|:--------------|------------:|-------------------:|:--------------|------------:|-------------------:|:---------------------------|:---------------------------|
| XG2      | S2: 0.9 + 0.3                                       |    0.40 | S2 p_A 0.9    |        1.09 |               1.00 | S2 p_A 0.6    |        1.21 |               1.28 | S2 p_A 0.3    |        1.09 |               1.07 | 0.13 / 0.52                | 1.56 / 1.98                |
| XG2      | S2: 0.9 + 0.6 + 0.3                                 |    0.41 | S2 p_A 0.9    |        1.04 |               1.00 | S2 p_A 0.6    |        1.14 |               1.19 | S2 p_A 0.3    |        1.06 |               0.98 | 0.14 / 0.22 / 0.62         | 1.70 / 1.87 / 2.18         |
| XG2      | S2: 1.0 + 0.0 (sorted A / sorted B, purity unknown) |    0.46 | S2 p_A 1.0    |        1.22 |               1.18 | S2 p_A 0.5    |        1.40 |               1.67 | S2 p_A 0.0    |        1.11 |               0.93 | 0.14 / 0.41                | 1.60 / 2.71                |
| XG2      | S1: 0.9 + 0.3                                       |    0.70 | S1 p_A 0.9    |        1.10 |               1.07 | S1 p_A 0.6    |        1.22 |               1.35 | S1 p_A 0.3    |        1.08 |               1.06 | 0.10 / 0.12                | 1.08 / 1.22                |
| XG2      | S4: 0.9 + 0.3                                       |    0.50 | S4 p_A 0.9    |        1.18 |               1.04 | S4 p_A 0.6    |        1.20 |               1.21 | S4 p_A 0.3    |        1.04 |               1.09 | 0.25 / 0.39                | 3.45 / 3.99                |
| XG2      | p_A 0.7: S1 + S4                                    |    0.69 | S1 p_A 0.7    |        0.62 |               0.61 | S4 p_A 0.7    |        3.02 |               2.94 | –           |      –    |             –    | 0.15 / 0.31                | 1.14 / 2.94                |
| R1t      | S2: 0.9 + 0.3                                       |    0.49 | S2 p_A 0.9    |        0.97 |               1.08 | S2 p_A 0.6    |        1.75 |               2.15 | S2 p_A 0.3    |        2.49 |               2.33 | 0.03 / 0.05                | 0.14 / 0.16                |
| R1t      | S2: 0.9 + 0.6 + 0.3                                 |    0.50 | S2 p_A 0.9    |        0.84 |               0.95 | S2 p_A 0.6    |        1.55 |               1.81 | S2 p_A 0.3    |        2.16 |               2.00 | 0.04 / 0.05 / 0.06         | 0.15 / 0.11 / 0.18         |
| R1t      | S2: 1.0 + 0.0 (sorted A / sorted B, purity unknown) |    0.60 | S2 p_A 1.0    |        1.09 |               1.37 | S2 p_A 0.5    |        3.56 |               4.37 | S2 p_A 0.0    |        4.53 |               2.55 | 0.03 / 0.05                | 0.13 / 0.25                |
| R1t      | S1: 0.9 + 0.3                                       |    0.74 | S1 p_A 0.9    |        1.04 |               1.10 | S1 p_A 0.6    |        1.59 |               1.95 | S1 p_A 0.3    |        2.49 |               2.36 | 0.03 / 0.04                | 0.14 / 0.15                |
| R1t      | S4: 0.9 + 0.3                                       |    0.33 | S4 p_A 0.9    |        1.02 |               1.19 | S4 p_A 0.6    |        2.08 |               2.62 | S4 p_A 0.3    |        2.36 |               1.99 | 0.04 / 0.06                | 0.15 / 0.16                |
| R1t      | p_A 0.7: S1 + S4                                    |    0.46 | S1 p_A 0.7    |        0.71 |               0.66 | S4 p_A 0.7    |        2.50 |               2.39 | –           |      –    |             –    | 0.04 / 0.05                | 0.11 / 0.10                |

- **Two conditions with different p_A help, but modestly on the late XG2 endpoints:**
  enriched 0.9 + depleted 0.3 (mostly single seeding) vs one condition at the mean p_A 0.6:
  λ_min ×1.21, Var(B→A) ÷1.28; vs the enriched condition alone ×1.09. Three conditions
  (0.9/0.6/0.3) are not better than two. Sorted A + sorted B with unknown purity (1.0 + 0.0):
  ×1.22 vs A-sorted alone, ×1.40 vs a 50/50 single condition.
- **With early time points the gain is large against a mixed single condition** (R1: ×1.75 vs
  p_A 0.6, ×2.5 vs p_A 0.3, sorted A + sorted B ×3.6 vs 50/50) **but not against the
  enriched condition alone** (×0.84–1.09): splitting clones from an A-enriched condition into
  a depleted one does not pay; what pays is the contrast when the alternative is a single
  mixed population.
- **The founder parameters per condition**: the enriched condition's p_A is identified on XG2
  for T truths (sd 0.13), the depleted one's is not (0.52); P truths: neither (sd > 1). With
  early time points both are identified (0.03/0.05 T, 0.14/0.16 P).
- **Mixing sizes across conditions does not help** (p_A 0.7, half single-cell + half uniform
  1–5 vs all single-cell: ×0.62 XG2, ×0.71 R1): every clone from the multi-cell condition is
  simply less informative.
- Estimation still costs in combined designs: E/K λ_min 0.40–0.70 (XG2), 0.33–0.74 (R1).


### 3. Simulation check at n = 512 per endpoint (NUTS dataset 0, MLE datasets 0–11)

Figure `figures/nuts_check.png` (NUTS 90% intervals, known vs estimated, with the 12 MLEs);
tables `tables/nuts_runs.{csv,md}`, `tables/mle_summary.{csv,md}`, `tables/mle_datasets.csv`,
`tables/progenitor.{csv,md}`, `tables/progenitor_reliability.json`, `tables/multi_mle.md`.

**Setup.** Data from the misspecification simulator extended to founder sizes 1–5
(`run.py`: m ~ π, each founder A with probability p_A; true founders recorded), 512 detected
clones per generation 4/6/8/10, exact truths (T1 with B divisions exactly 0). Models:
**est** = p_A and the size probabilities on the true support free (flat priors; S1: sizes {1},
p_A free), **known** = founder rows fixed at the truth. NUTS as the free-founder study
(adjoint kernel, 4 chains from the best of 8 L-BFGS optima, 300 warmup, tree depth ≤ 7,
ESS/R-hat stopping rule) but at most **400** draws per chain. MLE: L-BFGS from the truth,
4 prior draws and a "p_A high, B→A low" start (the parent-swap branch); best optimum.

**Scope actually run** (reduced; local machine only, ≤ 8 threads while the convergence
experiment ran): NUTS 16 runs = T1_ref and P3 × {S1/1.0, S2/0.7, S3/0.9} × {est, known},
plus T1_ref S4/0.5 and T3_Bdiv S2/0.7 (est, known). Not run: P3 S4/0.5, T3 on S1/S3/S4,
NUTS datasets ≥ 1. MLE: 3 truths (T1, T3, P3) × 4 distributions × {est, known} × 12
datasets (288 fits).

**B→A, m_BA, ρ (MLE, 12 datasets; RMSE; in brackets the fraction of datasets with
|B→A error| > 0.3, i.e. the fit on the wrong branch):**

| truth | founders | B→A RMSE est | B→A RMSE known | m_BA RMSE est / known | ρ RMSE est / known | Fisher sd B→A est / known |
|---|---|---|---|---|---|---|
| T1 | S1, p_A 1.0 | 0.09 (0%) | 0.18 (8%) | 0.10 / 0.06 | 0.004 / 0.006 | 0.07 / 0.07 |
| T1 | S2, p_A 0.7 | 0.26 (8%) | 0.12 (8%) | 0.10 / 0.09 | 0.008 / 0.008 | 0.12 / 0.08 |
| T1 | S3, p_A 0.9 | 0.29 (25%) | 0.08 (0%) | 0.18 / 0.09 | 0.005 / 0.004 | 0.14 / 0.11 |
| T1 | S4, p_A 0.5 | 0.37 (25%) | 0.12 (0%) | 0.18 / 0.06 | 0.009 / 0.005 | 0.20 / 0.10 |
| T3 | S1, p_A 1.0 | 0.21 (8%) | 0.14 (8%) | 0.10 / 0.06 | 0.005 / 0.005 | 0.10 / 0.09 |
| T3 | S2, p_A 0.7 | 0.20 (8%) | 0.07 (0%) | 0.13 / 0.12 | 0.007 / 0.005 | 0.16 / 0.10 |
| T3 | S3, p_A 0.9 | 0.26 (25%) | 0.27 (25%) | 0.22 / 0.17 | 0.007 / 0.006 | 0.19 / 0.13 |
| T3 | S4, p_A 0.5 | 0.30 (42%) | 0.15 (0%) | 0.31 / 0.19 | 0.007 / 0.006 | 0.25 / 0.13 |
| P3 | S1, p_A 1.0 | 0.16 (0%) | 0.12 (0%) | 0.19 / 0.14 | 0.008 / 0.007 | 0.18 / 0.13 |
| P3 | S2, p_A 0.7 | 0.12 (0%) | 0.14 (0%) | 0.24 / 0.24 | 0.010 / 0.009 | 0.26 / 0.21 |
| P3 | S3, p_A 0.9 | 0.12 (0%) | 0.12 (0%) | 0.29 / 0.35 | 0.006 / 0.005 | 0.32 / 0.19 |
| P3 | S4, p_A 0.5 | 0.19 (0%) | 0.16 (0%) | 0.35 / 0.35 | 0.007 / 0.003 | 0.48 / 0.48 |

- **Known founder distribution: the transitions are recovered about as well as with single
  A founders** (B→A RMSE 0.07–0.16, except T1 S1 0.18 and T3 S3 0.27, each from one to three
  datasets on the wrong branch); multi-cell seeding with known sizes costs little here.
- **Estimated: B→A becomes 2–4× worse for the T truths with multi-cell seeding** (T1: 0.26 /
  0.29 / 0.37 for S2 / S3 / S4 vs 0.12 / 0.08 / 0.12 known), with 25–42% of the datasets on
  the wrong branch (B→A ≈ 0 with p_A ≈ 1 or p_A ≈ 0) for S3/S4; the bias is downward (mean
  −0.11 to −0.22). This is larger than the asymptotic Fisher ratio (E/K variance 0.3–0.5):
  at n = 512 the cost of estimating the founders is mostly *branch selection along the
  parent-swap ridge*, not local spread. For P3 (small B→A) estimating costs little.
- **Single cells with p_A unknown (S1, est)** is not worse than known on these truths
  (T1 0.09 vs 0.18: the free p_A sometimes rescues a dataset that the all-A model fits on
  the B→A ≈ 0 branch).
- **ρ** is unaffected (RMSE 0.003–0.010 everywhere, ≈ the Fisher sd); **m_BA** loses less than
  B→A (RMSE est / known 1–2×, 3× only for T1 S4).
- **Founder parameters (MLE):** the mean founder size is recovered (RMSE S2 0.04 for P3/T3,
  0.22 for T1; S3 0.10–0.16; S4 0.14–0.37; truths 1.44 / 2 / 3); p_A is not (RMSE 0.06 only for
  T1 S1, otherwise 0.20–0.63, individual MLEs at 0 or 1), matching the Fisher picture
  (sd(p_A) 0.07–2.2 for these cases).
- **Fisher vs observed spread.** Where the posterior is unimodal the Fisher sd predicts it:
  P3, Fisher sd including the flat prior (Gaussian approximation, `fisher_sd_with_prior` in
  `nuts_runs.csv`) 0.069–0.095 vs posterior sd 0.058–0.081 for B→A, 0.10–0.16 vs 0.11–0.16
  for m_BA; ρ in every run (0.004–0.010, within 20%). For T1/T3 the posterior sd of B→A is
  1.1–3× the Fisher sd (ridge, two branches), and MLE RMSEs of the estimated model exceed
  the Fisher sd by 1.2–2.2× for the same reason.

**NUTS (dataset 0).** Coverage of the 90% intervals for B→A: known 6/8 (misses: T1 S1/1.0 —
the standard XG2 model itself is on the B→A ≈ 0 branch on this dataset, 0.09 [0.00, 0.25] —
and T1 S4/0.5, 0.61 [0.40, 0.77]); estimated 3/8 (P3 3/3; T1 S1/S2/S3/S4 and T3 S2 missed:
posterior means 0.28 / 0.54 / 0.34 / 0.33 vs 0.8 and 0.37 vs 0.6, z −2.1 to −3.5: the
downward ridge bias of the free-founder study). m_BA is covered in 14/16 runs, ρ in 14/16.
p_A (est): covered 2/8 (posterior means 0.22–0.64 for truths 0.5–1.0; intervals span most of
[0, 1]); mean founder size covered 5/6 (missed: T1 S3/0.9, 2.37 [2.07, 2.69] vs 2.0).

**Mixing.** 11/16 runs are flagged (rank R-hat > 1.05, bulk ESS < 100 or chain means > 0.5 sd
apart): all 4 S1 runs (R-hat 1.03–1.21, chain spread 0.57–0.90), P3 S2 known, P3 S3 est and
known (R-hat 1.26, ESS 12), T1 S3 known, T1 S4 est, T3 S2 est and known. No run met all
stopping targets within 400 draws per chain (all stopped at max_draws). Poor mixing occurs
with known founders too (P3, T1 S1): it is the B-row ridge of the XG2 design, not only the
founder parameters. Read the flagged runs as indicative. Run time per job (4 threads):
9–17 min (S1), 31–158 min (S2), 25–93 min (S3), 81–98 min (S4); one gradient costs 0.01 s
(S1) to 0.03–0.12 s (sizes up to 5: larger grids).

**Per-clone progenitor posterior** (S2–S4 runs; NUTS average over 50 draws and MLE plug-in;
oracle = at the true parameters):

| task | AUC (fit = oracle) | Brier fit (known / est) | Brier constant |
|---|---|---|---|
| multi-founder clone (m > 1) | 0.86–0.94 | 0.08–0.13 / 0.08–0.17 | 0.15–0.22 |
| all founders A | 0.53–0.73 | 0.14–0.23 / 0.15–0.50 | 0.15–0.24 |
| founder count (corr. of posterior mean with the true m) | 0.71–0.83 | | |

- **The founder count of a clone is recoverable** (AUC 0.86–0.94 for "more than one founder";
  the posterior-mean count correlates 0.71–0.83 with the truth) and well calibrated when the
  sizes are known; with estimated sizes it is over-predicted where the mean size is
  over-estimated (T1 S2 est: predicted 0.29 / 0.50 / 0.69 vs observed 0.13 / 0.28 / 0.44).
- **The founder types are not** (AUC 0.53–0.73, as in the free-founder study); with p_A
  estimated the composition probabilities are badly calibrated (Brier up to 0.50, worse than
  a constant) because p_A is mis-estimated (mean predicted P(all A) 0.15–0.31 for true rates
  0.62–0.81 in the S2/S3 NUTS runs).

### 4. Several conditions: joint MLE (`multi.py`)

T1_ref and P3, 12 datasets: two conditions S2/p_A 0.9 and S2/p_A 0.3 (256 clones per endpoint
each), shared transitions, each condition's p_A and sizes estimated (joint likelihood = sum
of the two exact likelihoods), vs one S2/p_A 0.6 condition with 512 per endpoint (same start
strategy as the main MLE):

| truth | setup | B→A RMSE (bias) | m_BA RMSE | ρ RMSE | p_A RMSE per condition | mean size (truth 1.44) |
|---|---|---|---|---|---|---|
| T1 | 0.9 + 0.3 | 0.083 (−0.03) | 0.085 | 0.006 | 0.27 / 0.21 | 1.58 / 1.53 |
| T1 | 0.6 single | 0.152 (−0.09) | 0.082 | 0.009 | 0.28 | 1.47 |
| P3 | 0.9 + 0.3 | 0.096 (−0.00) | 0.154 | 0.014 | 0.45 / 0.41 | 1.45 / 1.44 |
| P3 | 0.6 single | 0.155 (−0.08) | 0.304 | 0.008 | 0.38 | 1.44 |

Two contrasting conditions reduce the B→A RMSE by ~40% for both truths (MSE ÷3.3 / ÷2.6;
the Fisher prediction for the XG2 design is ÷1.28 median: again the finite-sample gain is
mostly fewer fits on the wrong branch, 0/12 vs 1/12 for T1) and halve the m_BA RMSE for P3;
ρ is similar (T1 0.006 vs 0.009, P3 0.014 vs 0.008). The per-condition p_A stays poorly
determined (RMSE 0.2–0.45), as predicted for the XG2 endpoints.

**NUTS with several conditions is not supported by the pipeline**: `ExactLikelihood` takes one
founder distribution for all cohorts, and `ExactGW.run` uses the likelihood's parameter space
(`lik.space`) for initialisation, decoding and diagnostics. Proposal: a
`MultiConditionLikelihood` holding one `ExactLikelihood` per condition, with theta = (shared
outcome-row sticks, condition-1 founder sticks, condition-2 founder sticks, …),
`log_likelihood` = sum (the gradient blocks map directly; `multi.Joint` already does this),
`prior_a` / `prior_b` concatenated, and a combined `ParameterSpace` (rows `founder@c`,
`founder_size@c`) so that `decode`, the L-BFGS initialisation and the diagnostics work
unchanged; in the config, a list of cohorts per condition with condition-specific
`law.fixed.founder` / `law.founder_sizes`.

## Caveats

- Fisher numbers are asymptotic (local, Gaussian). At n = 512 per endpoint the estimated
  founder model is dominated by the parent-swap ridge (branches), so the finite-sample cost of
  estimating the founders is larger than the Fisher ratio for the T truths (Section 3).
- The boundary truths T1/T2/T4 use the eps = 0.01 variant in the Fisher calculation
  (B-founded information is non-regular at B divisions = 0); the simulations use the exact
  truths.
- p_A = 1, p_A = 0 and the S1 sizes are boundary values; the E numbers use the two-sided
  information and the true size support. Estimating sizes outside the true support (e.g.
  π_2…π_5 free when every clone is single) is non-regular and not covered by the Fisher part.
- Founder types are i.i.d. per founder cell and founder sizes independent of type (as in the
  model). Clumps of sorted cells, or size-dependent purity, are not modelled.
- NUTS: one dataset per case, at most 400 draws per chain, 11/16 runs poorly mixed.

## Files

- Scripts (`scripts/founder_distribution/`): `fd_common.py` (founder distributions, designs,
  functionals, Schur complement), `fisher.py` (information cache), `validate_fisher.py`,
  `analyse_fisher.py` (tables incl. multiple conditions), `figures.py` (Fisher figures),
  `run.py` (simulator with sizes 1–5, NUTS and MLE jobs), `multi.py` (joint MLE of several
  conditions), `analyse_sim.py` (NUTS / MLE / progenitor tables and `nuts_check.png`),
  `queue.sh` (job queue as run).
- Results (`/home/matteo/BIO-SBI/results/exact_gw/founder_distribution/`):
  `fisher/<truth>.npz` (15 × 15 information per distribution and generation),
  `nuts/<dist>/<model>/<truth>_s0.{json,npz}` (posterior draws, per-clone progenitor
  posterior, true founders), `mle/…` (12 datasets), `multi/…`, `logs/`;
  `old/multi_localstarts` = a first multi-condition run whose random starts failed (only the
  truth start converged; superseded).
- Figures (`figures/`): `info_lam.png`, `info_btoa.png`, `info_mba.png`, `info_rho.png`,
  `cost_per_truth.png`, `founder_identifiability.png`, `multi_conditions.png`,
  `nuts_check.png`.
- Tables (`tables/`, CSV + md): `fisher_key`, `fisher_summary`, `fisher_main`,
  `fisher_per_truth_{xg2,r1t}`, `multi_conditions{,_summary,_ratios}`,
  `fisher_validation.json`, `mle_summary`, `mle_datasets`, `multi_mle`, `nuts_runs`,
  `progenitor`, `progenitor_reliability.json`.
