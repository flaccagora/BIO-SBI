# Exact Bellman–Harris endpoint law (`bhexact.py`): validation and Fisher information

## Model and method
The model is confirmed from `src/bio_sbi/simulators/bellman_harris.py` (`bh_forward`). Each cell of type k lives
Gamma(m, mean_k/m). At the end of its life it draws one outcome from row k and is replaced by
that outcome's daughters, all born at that instant with age 0. `persist` gives one new cell of the
same type with a fresh lifetime, `to_other` gives one new cell of the other type, and `death`
gives nothing. The single A founder is born at t = 0 with age 0. Counts are the cells alive at
the horizon (lifetime end ≥ t). The repo's B row is ordered death, **to_A, persist**, …;
`from_repo` reorders it to the gwexact order (B: death, persist, to_A, …).

Method of stages: m Exp(λ_k = m/mean_k) phases. On the grid there are 2m complex backward
Kolmogorov ODEs per point (equations in the module docstring), and the clone pgf is
F = G_{A,1}. The ODEs are integrated with **fixed-step DOP853 (8th order, scipy tableau), dt = 1 h**,
in a numba kernel limited to 2 threads. The grid is rectangular L_A × L_B, and conjugate
symmetry halves it. Then p = real(fft2 F)/(L_A L_B). Forward-mode tangents cover 12
polynomial-extension P directions plus d/dlog mean_A and d/dlog mean_B (`law_bh_jac`). A
non-integer shape raises `NotImplementedError`, which names the BH integral-equation route.

Truth: reversible_ab/two_type reference rows (A 0.03/0.03/0.05/0.55/0.05/0.29; B death 0.1,
persist 0.1, to_A 0.8, no B divisions), A mean 20 h, B mean 30 h, shape 8.
Table size **L = (L_A, L_B) = (1024, 512)**.

Scripts: `bhexact_validate.py` (items a–f), `bhfisher.py` (derivatives and Fisher information).

## Validation (all pass)
**(a) Normalisation, negative mass and tail.** Tail mass is computed from 1-D marginal pgfs on N = 8192 points.

| h | Σp − 1 | min p | total negative mass | P(A ≥ 1024) | P(B ≥ 512) | p00 |
|---|---|---|---|---|---|---|
| 96 | 0 | −5e-18 | 2.8e-14 | <1e-15 | <1e-15 | 0.04466 |
| 144 | −1e-16 | −4e-18 | 2.8e-14 | <1e-14 | <1e-14 | 0.04689 |
| 240 | −1e-16 | −1e-18 | 2.1e-14 | 9.4e-13 | 4.6e-9 | 0.04749 |

Marginal tails at 240 h for choosing L: P(A ≥ 512) = 2.1e-5, P(A ≥ 768) = 6.0e-9,
P(B ≥ 256) = 7.1e-4. At 144 h, L = 256 is enough (P(A ≥ 128) = 4.6e-11).

**(b) Means vs Monte Carlo** from the repo's `bh_forward` (2e5 clones per horizon).

| h | exact E[A], E[B] | MC ± se | z |
|---|---|---|---|
| 96 | 4.6282, 3.0146 | 4.6162 ± 0.0078, 3.0112 ± 0.0057 | −1.55, −0.59 |
| 144 | 12.4869, 8.1372 | 12.4950 ± 0.0205, 8.1294 ± 0.0140 | 0.39, −0.56 |
| 240 | 90.922, 59.251 | 90.995 ± 0.147, 59.312 ± 0.096 | 0.50, 0.63 |

Extinction probability, exact vs MC: 0.04466 vs 0.04419; 0.04689 vs 0.04705; 0.04749 vs 0.04774.

**(c) Full 2-D law vs Monte Carlo** (2e5 clones). The TV reference distribution comes from a
parametric bootstrap: 200 multinomial samples of the same size drawn from the exact law. The
chi-square pools cells with expected count < 5.

| h | TV | bootstrap TV mean (q95) | bootstrap p | χ² / dof | χ² p |
|---|---|---|---|---|---|
| 96 | 0.01140 | 0.01149 (0.01266) | 0.53 | 312.7 / 307 | 0.40 |
| 144 | 0.02708 | 0.02569 (0.02693) | 0.02 | 1285.8 / 1154 | 0.004 |
| 240 | 0.1127 | 0.1123 (0.1135) | 0.25 | 8349.7 / 8289 | 0.32 |

I re-ran the 144 h result with 3 fresh seeds × 4e5 clones each, at 120, 144 and 168 h. The
χ² p-values were 120 h: 0.12, 0.72, 0.89; **144 h: 0.96, 0.52, 0.71**; 168 h: 0.90, 0.88, 0.14. No
cell had a standardized residual above 4.1 among ~1400–2500 cells. The first 144 h draw was a
chance fluctuation.

Single cells at 96 h (exact vs MC ± se): P(1,0) 0.016230 vs 0.01648 ± 0.00028;
P(0,1) 0.009009 vs 0.00933 ± 0.00021; P(2,0) 0.028191 vs 0.02787 ± 0.00037 (all within 1.5 se).

**(d) shape = 1 means vs `BellmanHarris.expected_counts`** (closed form, matrix exponential).
Relative error is ≤ 5e-12 at all three horizons. At 240 h: 340.570756438 vs 340.570756438 (A)
and 209.114106328 vs 209.114106329 (B). Tail mass at N = 16384 is < 1e-13.

**(e) Step size** (2-D tables, dt = 1 h). Halving dt changes p by at most 9.3e-16, 2.1e-16 and 1.0e-17
(96, 144, 240 h). Doubling to dt = 2 h changes it by ≤ 2.1e-13. For comparison, RK4 needs
dt ≈ 0.06 h to reach the same level (RK4 with dt = 0.25 h vs 0.125 h differs by 2.7e-10).

**(f) Timing** (2 threads, nice 10, machine at load ≈ 19/20). One likelihood evaluation at 240 h:
L 1024×512: 53 s; 768×512: 40 s; 512×512: 27 s. The 144 h and 240 h tables come from one
trajectory, so {144, 240} costs the same 53 s. Cost is linear in grid points × steps. The
recommended L is (768, 512): missed mass 6e-9, about 40 s per evaluation on 2 threads, so
roughly 4 s with 20 threads. dt = 2 h halves the time at ≤ 2e-13 error.

## Derivatives
`law_bh_jac` propagates 14 tangents, covering λ_k via d/dlog mean_k = −λ_k. It was checked
against central finite differences (eps = 1e-5, L = 48, horizons 48 and 96 h). The maximum
absolute error over the tables is 6e-12 to 3.2e-10, against maximum |derivative| values of 0.12–1.2.
That is the O(eps²) truncation size, so the derivatives pass. A Jacobian at L = 768×512 for
{144, 240} h took 511 s.

## Fisher information at the BH truth, real-like design {144 h: 165, 240 h: 101}, detected clones
**Coordinates.** Probabilities use "drop-death" free coordinates: row entries other than death,
with death = 1 − sum (tangent directions e_j − e_death, the gwexact `tangent_basis` convention).
Lifetimes use log mean_A and log mean_B. The CR sds of single entries and of the B-division mass
are invariant to which entry is dropped. Eigenvalues are not, so the table also gives the
condition number of the unit-diagonal (correlation-scaled) matrix. The missed conditional mass
is 7e-11 at 144 h and 1e-9 at 240 h.

| | reversible_ab (7 + 2) | two_type (10 + 2) |
|---|---|---|
| λ_min (total I) | 0.585 | 0.211 |
| λ_max | 9.14e4 | 1.02e5 |
| condition number | 1.6e5 | 4.8e5 |
| scaled λ_min / condition number | 1.4e-4 / 3.8e4 | 3.3e-5 / 2.3e5 |
| CR sd A.div_AA | 0.229 | 0.536 |
| CR sd A.mean_lifetime (log; hours) | 0.191 (3.8 h) | 0.396 (7.9 h) |
| CR sd B.to_A | 0.817 | 0.987 |
| CR sd B-division mass | — | 0.984 |
| n for sd < 0.05: A.div_AA | 5.6e3 | 3.1e4 |
| n for sd < 0.05: A log-lifetime | 3.9e3 | 1.7e4 |
| n for sd < 0.05: B.to_A | 7.1e4 | 1.04e5 |
| n for sd < 0.05: B-division mass | — | 1.03e5 |

"n" means clones in the same 165:101 split, from n = 266 × (sd/0.05)².

Smallest-eigenvalue directions:
- reversible_ab: B.persist −0.52, B.to_A +0.58, B.log_mean +0.50, A.persist −0.30. B
  transitions trade off against B's lifetime.
- two_type: B.log_mean −0.69, B.persist +0.46, B.div_AB −0.41, B.to_A −0.24, B.div_BB +0.20.

**Conclusion.** Under the exact BH likelihood the Fisher matrix is **nonsingular** for both laws
(λ_min > 0, far above the ~1e-10 accuracy of the derivatives). The BH self-renewal/lifetime
ridge and the BH B-row problem are therefore **practical, not structural**. At 266 detected
clones the local CR sds are of the order of the prior range for B.to_A and the B-division mass
(≈ 1), and 0.2–0.5 for A.div_AA. Getting below 0.05 would take about 4e3–3e4 clones for the A
parameters and about 1e5 for the B row.

Caveats: the CR bounds are local. At n = 266 they exceed the simplex width, so the posterior is
prior-dominated there and the bound only describes the large-n regime. For two_type the truth
sits on the boundary (B divisions = 0), where the Fisher matrix is the one-sided local
information.
