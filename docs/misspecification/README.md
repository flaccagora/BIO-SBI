# Misspecification study: exact-likelihood two-type GW on organoid clone counts

**Question.** The exact GW analysis assumes: every clone starts from **one LGR5⁺ (A) cell**,
cell types are **scored without error**, every clone follows the **same** outcome
probabilities, divisions are **synchronous** generations, rates do **not depend on clone
size or time**, and **every cell is counted**. What happens to the estimates (especially
B→A, dedifferentiation), the intervals and the dedifferentiation test when the data come
from biology or an assay that breaks one of these assumptions? Can the problem be seen in
the data, and does modelling it fix it?

**Status (2026-11-23 06:30): complete.** n = 512 clones per endpoint, generations 4/6/8/10,
all 9 XG2 truths, 10 datasets per scenario × truth (MLE, goodness of fit, extended models),
profile intervals on 3, the dedifferentiation LRT on 100 null datasets per scenario
(nulls N1, N2), NUTS (flat prior) on 1 dataset per truth for F70, MTba2 and HET10.
Run time: 00:45–06:30 on ≤ 16 local workers. Tables: `tables/`, figures: `figures/`;
rerun the analysis with `uv run python scripts/misspecification/analyse.py`.

## Results

| scenario | B→A bias (mean / max over truths) | ρ bias | B→A 95% cover | dediff FPR (N1/N2) | detected: GOF / extended LRT | modelled: max B→A bias, cover, FPR |
|---|---|---|---|---|---|---|
| correctly specified | −0.04 / 0.14 | −0.1% | 0.78 | 0.03 / 0.04 | 6% (size) | — |
| founders A/B 70/30 | −0.18 / 0.65 | −0.4% | 0.63 | 0.05 / 0.03 | 11% / 58% | 0.06, 0.89, 0.05/0.02 |
| 15% doublets (A) | −0.28 / 0.75 | +1.8% | 0.26 | 0.19 / 0.17 | 51% / 100% | 0.13, 0.93, 0.04/0.07 |
| 15% doublets, founders 70/30 | −0.30 / 0.79 | +1.5% | 0.15 | 0.16 / 0.05 | 57% / 100% | 0.09, 0.89, 0.04/0.02 |
| B read as A 0.5% / 2% / 5% | ≈ base | ≈ 0 | 0.78–0.96 | **0.98–1.00** | 4–7% / 9–10% | ≈ base, 0.78–0.93, 0.00–0.09 |
| symmetric mis-typing 2% | ≈ base | −0.2% | 0.81 | **1.00** | 8% / 9% | ≈ base, 0.81, 0.03/0.06 |
| heterogeneity κ = 50 | −0.29 / 0.76 | +1.7% | 0.22 | 0.45 / 0.28 | 82% | (no extended model) |
| heterogeneity κ = 10 | −0.30 / 0.65 | +7.5% | 0.00 | **1.00** | 100% | |
| asynchrony ±1 generation | −0.34 / 0.78 | +1.9% | 0.07 | 0.12 / 0.11 | 69% | |
| density dependence | +0.10 / 0.25 | −7.2% | 0.15 | 0.00 / 0.05 | 100% | |
| cell loss 10% / 20% | −0.06 / 0.30–0.45 | −0.2/−0.5% | 0.81 | 0.02–0.04 | 9% / 14–52% | 0.20–0.23, 0.81 |
| B→A doubles after gen. 5 | +0.01 / 0.13 | −0.6% | 0.96 | 0.02 / 0.05 | 27% | |
| growth slows after gen. 5 | −0.08 / 0.37 | −6.7% | 0.52 | 0.14 / 0.19 | 78% | |

FPR: false-positive rate of the χ̄² dedifferentiation LRT at nominal 5% on data with no
dedifferentiation. GOF: binned G² goodness of fit with a threshold calibrated on correctly
specified data (size 6%). Extended LRT: standard model vs the model with the relevant
founder / confusion / capture parameter estimated (size 0–7% on correctly specified data).
"Modelled": the extended model with the true founder mixture, confusion matrix or capture.

**Main findings**
- **Mis-typing is invisible and fatal for the dedifferentiation test.** Even 0.5% of
  KRT20⁺ cells scored LGR5⁺ gives 98–100% false positives, yet neither goodness of fit nor
  the extended-model LRT detects it (4–10%): a misread B cell is indistinguishable from a
  B cell that switched. It barely biases the transition estimates. Only an independently
  measured error rate fixes it (FPR back to 0–9%).
- **Unmodelled founders and doublets push B→A along the weak direction** (B→A traded for B
  divisions): for T1, T3, T4 B→A drops by 0.4–0.8. NUTS on 70/30 founders gives T1
  B→A = 0.02 ± 0.02 against 0.80 (z = −39): the posterior is confidently wrong. 70/30
  founders are hard to see (GOF 11%, extended LRT 58%); doublets are seen (51–100%).
  Estimating the founder mixture fixes the bias even though the mixture itself is weakly
  identified (estimates 0.24–0.81 for a true 0.7).
- **Organoid heterogeneity is the most damaging biology**: κ = 10 gives 0% coverage, ρ +7.5%
  and 100% dediff false positives; κ = 50 already 22% coverage and 28–45% false positives.
  It is detectable (82–100%) but the model has no fix yet.
- **Asynchrony and density dependence** bias B→A or m_BA and ρ by several sd and are
  detectable (69–100%); density dependence lowers ρ by 7%.
- **Cell loss** inflates the B→A error 1.5–1.9× and is nearly undetectable, but is harmless
  for the dediff test.
- **Identifiable summaries are robust to assay problems**: ρ and the stable A fraction move
  by < 2% under founders, mis-typing and cell loss; they move under biology (heterogeneity,
  density, time-varying growth: 6–8% in ρ).

**For the lab**: sort single cells (avoid doublets) and record founder identity; measure
the KRT20⁺→LGR5⁺ mis-scoring rate independently before any dedifferentiation claim;
check for organoid-to-organoid heterogeneity (replicate wells, overdispersion of clone
sizes) and for time/size-dependent growth; match time points to division counts.

## Design

**Analysis model (always the same unless stated):** two-type Galton–Watson, 12 outcome
probabilities (10 free), one A founder per clone, perfect typing, every cell counted, all
clones identical, synchronous generations, constant rates; exact likelihood (adjoint
kernel) conditioned on detection.

**Data:** n = 512 detected clones per endpoint, generations {4, 6, 8, 10} (independent
cohorts), the 9 XG2 truths (T1_ref, T2_toA05, T3_Bdiv, T4_slowA, T5_C7, P1–P4; vectors in
`scripts/misspecification/common.py`). Clones with no observed cell are not detected and
are excluded, as in the pipeline.

| scenario | what the data do that the model does not |
|---|---|
| base | nothing (correctly specified reference on the same seeds) |
| F70 | each clone founded by one cell, A with probability 0.7, B with 0.3 |
| D15A | 15% of clones founded by two A cells (doublets / aggregates) |
| D15mix | 15% of clones founded by two cells; every founder A/B 70/30 (F70 + doublets) |
| MTba0.5 / MTba2 / MTba5 | each B (KRT20⁺) cell scored as A (LGR5⁺) with probability 0.5% / 2% / 5% |
| MTsym2 | each cell scored as the other type with probability 2% |
| HET50 / HET10 | each clone draws its own outcome rows from Dirichlet(κ·θ), κ = 50 / 10 (organoid-to-organoid fate heterogeneity) |
| ASYNC | a clone observed at nominal generation h has gone through h−1, h, h+1 generations with probability 0.25 / 0.5 / 0.25 |
| DENS | each cell's division probabilities × 1/(1 + N/K) (N = current clone size, lost mass → persist); K per truth so the mean clone size at generation 10 is halved (K = 28–117) |
| LOSS90 / LOSS80 | each cell observed independently with probability 0.9 / 0.8 |
| TVtoA | B→A doubles from generation 6 on (capped at (1 + B→A)/2; other B outcomes rescaled) |
| TVslow | division mass × 0.75 from generation 6 on (lost mass → persist) |

The mis-typed datasets are the base datasets of the same seed relabelled (paired); the other
scenarios are independent draws with the same seed index.

**Fits per scenario × truth** (`scripts/misspecification/run.py`):
- **MLE** of the standard model on datasets 0–4: multistart L-BFGS-B (generic start, 6 flat
  Dirichlet starts, B→A pinned at 0.02 / 0.3 / 0.7 then released; the two best polished with
  ftol 1e-15 / gtol 1e-9). Never started at the truth.
- **Profile-likelihood 95% intervals** (drop 1.92 nats, χ²₁) for B→A and m_BA on datasets
  0–2 (plib constrained model, warm-started outward steps + secant refinement).
- **Where the exact likelihood can represent the scenario** (founder mixture and founder
  size, confusion matrix, capture probability are all supported by the likelihood, just not
  as free parameters in the pipeline): the same fit with the true setting known ("fixed"),
  and the **extended model** with the setting estimated (MLE by a grid over the extra
  parameter(s) — standard multistart MLE at each grid value — then a joint polish with a
  finite-difference gradient in the extra parameter). LR test extended vs standard with the
  boundary mixture ½χ²₀ + ½χ²₁ (¼χ²₀ + ½χ²₁ + ¼χ²₂ for founder type + doublet rate).
- **Goodness of fit:** binned G² of the observed (A, B) counts against the fitted law —
  per generation, 6 total-size classes (quantiles of the fitted law) × 5 A-fraction classes
  {0, (0, ⅓], (⅓, ⅔], (⅔, 1), 1}, bins with expected count < 5 pooled — summed over
  generations (≈ 40–60 df).
- **NUTS** (flat prior, 4 chains, adjoint kernel, adaptive stopping) of the standard model on
  dataset 0 of every truth for MTba2, F70 and HET10 (`scripts/misspecification/nuts.py`).
- **Dedifferentiation test** (`dediff_lrt.run_test`: LRT of B→A = B→AA = B→AB = 0, χ̄²
  calibration from the observed information) on data from the null rows N1 (B: death 0.2 /
  persist 0.8) and N2 (death 0.1 / persist 0.5 / B→BB 0.4) of the dediff study (A row =
  reference), n = 512, 50 datasets per scenario × null, every scenario; where representable
  also with the true setting modelled.

**Simulator.** The scenarios are simulated by a numpy two-type branching simulator
(`common.simulate`). Checked against the pipeline (`validate_simulator.py`,
`results/.../validation.md`): with 20,000 detected clones per generation for T1_ref, T3_Bdiv,
T5_C7 and P1, both this simulator and the pipeline's exact detection-conditioned sampler
agree with the exact law (binned G², no fitted parameters, p = 0.02–0.99) and with each
other (two-sample χ², p = 0.12–0.89); founder mixture + doublets, cell loss and symmetric
mis-typing agree with the exact law of the extended model (p = 0.05–0.94; the pipeline's own
founder-mixture sampler p = 0.46–0.80); clone heterogeneity (κ = 10) agrees with the
pipeline's random-effects law (`law.clone_heterogeneity`, p = 0.29 and 0.62).
