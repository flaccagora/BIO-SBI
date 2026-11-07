# Experiments plan

Goal: find out which parameters of the two-type (LGR5 = **A**, stem-like; KRT20 = **B**,
differentiated) branching model can be recovered from endpoint organoid counts, and
how robust that recovery is to the errors present in the real CRC0322 data (organoid
death, mis-typing / a third population, division-time variability, number of
organoids). Each result should end in a concrete recommendation: what to fix, what to
measure, how many organoids.

Original axes: **model (GW, BH) × inference (SBI method) × seed (dataset and inference)
× error combinations**. We do not run the full product. Every experiment below varies
one error axis against a fixed **baseline**, and the other axes are crossed only in the
experiments that need them.

---

## 0. Conventions (apply to every experiment)

### Baseline

| Axis | Baseline | Why |
|---|---|---|
| Law | `reversible_ab` (B never divides) | With `two_type`, `B.to_A` is not recovered at any budget (see E0) |
| Simulator | `galton_watson`, with `bellman_harris` as the second model | GW is cheap; BH is closer to the real data |
| Design (real-like) | GW `design.horizons=[6,10]`, BH `design.horizons=[144,240]`; `design.clones=[165,101]` | Day 6 and day 10 organoids in `submodules/GW/data` |
| Observation | `perfect` (the error experiments add noise on the **data** side) | |
| Summary | `moments` | `means` cannot see variance (E0 showed variance is what carries the information) |
| Inference | `npe`, 1 round | NLE/NRE only in X1 |
| Budget | `dev` (5k sims); `full` only to confirm a final result | 5k → 50k did not change recovery in E0 |
| Seeds | 5 (`seed=0..4`) | |

### What every run reports, and how results are read

* Per parameter: posterior mean, abs error, **contraction** (`1 - Var_post/Var_prior`),
  90% CI coverage across seeds.
* **SBC** (calibration averaged over the prior) and **PPC** `fraction_outside_95` (can
  the fitted model reproduce the data?).
* Always read the well-specified run next to the misspecified one at the same design.
  A **bias** means the posterior moves away from the truth under the error; an
  **information loss** means the contraction drops while the posterior stays unbiased.
* A parameter counts as *recovered* when contraction > 0.5, |error| < 0.05 and the
  truth lies inside the 90% CI in at least 4 of 5 seeds.
* Watch the prior tail: if the truth is improbable under the prior (as `B.to_A = 0.8`
  is under `Dirichlet(1,…)`), a flat posterior can look calibrated in SBC. Report the
  prior quantile of the truth.

### Layout

One directory per experiment, `results/<exp>/`, holding `info.md` (question, command,
commit, results, conclusion; same template as `results/b_row/info.md`), the sweep
subdirectories (`hydra.sweep.dir=results/<exp>/<sweep>`) and the
`bio-sbi-collect` tables.

### Cost (measured in E0)

GW + NPE: about 40 s per run at 5k simulations and about 5 min at 50k (4 threads),
including SBC on 200 cases. About 4 runs fit in parallel on the 20-core machine.
BH cost is still to be measured (P0).

---

## P. Prerequisites (code changes, before the experiments that need them)

- [ ] **P0 – BH timing**: `budget=smoke` and `budget=dev` with `simulator=bellman_harris`
      at the real-like design. Record the seconds per run here. **Not run yet** (machine
      busy); when it is idle, run one job at a time and read `total_seconds`,
      `training[*].simulation_seconds` and `training[*].training_seconds` from each
      `metrics.json` (or `uv run bio-sbi-collect results/p0_bh_timing`):
      ```bash
      uv run bio-sbi simulator=bellman_harris law=reversible_ab design=real_like \
          budget=smoke hydra.run.dir=results/p0_bh_timing/smoke
      uv run bio-sbi simulator=bellman_harris law=reversible_ab design=real_like \
          budget=dev hydra.run.dir=results/p0_bh_timing/dev
      # GW reference point at the same design and budget
      uv run bio-sbi simulator=galton_watson law=reversible_ab design=real_like \
          budget=dev hydra.run.dir=results/p0_bh_timing/gw_dev
      ```
      (`workers=4 torch_threads=4` are the defaults; note them with the timings.)
- [x] **P1 – separate seeds**: `data_seed` (default `${seed}`) seeds the truth draw
      (`"truth"`) and the reference dataset (data generator, `"reference"`); `seed` keeps
      training simulations, training, posterior sampling (`"ref"`), prior samples
      (`"prior-reference"`), PPC and SBC. Default reproduces earlier reference data
      bit for bit (regression test); `metrics.json` records both seeds.
- [x] **P2 – observed-organoid fraction**: `metrics.json` → `reference_detection` per
      horizon: `seeded`, `detected`, `detection_rate` (rejection loop, e.g. BH) and the
      exact `detection_probability` = 1 − P(no captured cell) for GW. Collected as
      `detection_rate_h<h>` / `detection_probability_h<h>`; training sims unaffected.
- [x] **P3 – new configs**: `observation/three_split.yaml` (top-level `q`),
      `law/three_type_traj.yaml` (C.to_A and C divisions with an A daughter fixed to 0;
      A and B rows free — assumption pending Q1), `design/real_like.yaml` (days 6/10,
      165/101 clones, GW generations or BH hours), and presets `e1a_founder_type`,
      `e1a_founder_number`, `e1b_death_detection`, `e1c_fixed_death`,
      `e2a_typing_known`, `e2b_typing_ignored`, `e2c_third_type`, `e3a_lifetime_free`,
      `e3a_lifetime_fixed`, `e3b_desync`, `e3c_persist`, `e4_design`, `x2_seeds`
      (commands and run counts in each file's header; every grid point composes and
      builds, tested).
- [x] **P4 – generic summariser**: `scripts/summarize.py <results/exp> [--params ...]
      [--group-by ...]` groups by sweep folder + varying overrides, averages over seeds,
      writes `summary_groups.csv` and `summary_group_parameters.csv` (mean, sd, error,
      contraction, coverage, `recovered`, SBC, PPC, detection).

---

## E0. Identifiability of B's row — DONE (`results/b_row/`)

Status: 30/30 runs done; direct checks done.

* With all transitions free (`two_type`), `B.to_A` (true 0.80) stays at its prior
  (mean ≈ 0.20, contraction ≈ 0) at 5k, 20k and 50k simulations and for every seed.
  SBC still looks calibrated, because the truth is in the prior tail.
* With B divisions fixed to 0 (`reversible_ab`), `B.to_A ≈ 0.70`, contraction ≈ 0.7.
* C2ST shows that switching and division differ only in clone-size variability. At
  32 clones per horizon only large differences (a ≥ 0.2) are detectable; at 512 clones
  a ≈ 0.05 is detectable.

Remaining:
- [x] Sequential NPE (`inference.method.rounds=4`) with `law=two_type` at 20k
      simulations (`results/b_row/two_type_snpe/`, 5/5 seeds): `B.to_A` 0.19–0.28,
      unchanged from amortised (0.18–0.31). Caveat: later rounds sample
      from the round-0 posterior, which is already prior-like for `B.to_A`, so SNPE
      alone cannot settle coverage vs identifiability.
- [x] `two_type` with a flat prior on `B.to_A` (`+experiment=b_row_flat_prior`,
      `results/b_row/two_type_flat_to_A/`): `B.to_A` 0.21 → 0.53 ± 0.21, truth covered in
      4/5 seeds, corr(`to_A`, `div_AA`) = −0.67. **Both** effects: the uniform-Dirichlet
      prior tail caused most of the miss, and the 3 × 32-clone data only weakly separate
      switching from division. SBC fails for B's *other* coordinates under this sparse
      prior (`B.to_A` itself is calibrated).
- [x] `results/b_row/info.md` finished (amortised, sequential, flat prior, conclusions).

---

## E1. Cell death

### 1a – Organoid death → initial conditions

*Assumption (from the biology): organoid loss changes only which clones start (number
and type of founder cells), not the growth dynamics.* We model this with the `founder`
(type mix) and `founder_size` (number of cells) rows.

Question: how stable is inference of the dynamics (A and B rows) when the initial
conditions are wrong?

| Sweep | Data | Inference |
|---|---|---|
| founder type | `'++data.law.fixed.founder={A: f, B: 1-f}'`, `f ∈ {1.0, 0.9, 0.7, 0.5}` | founder fixed to `{A: 1}` (wrong) / founder inferred (`'~law.fixed.founder'`) |
| founder number | `law@data.law=multi_founder`, `'data.law.reference.founder_size={"1": p1, ...}'`, `p1 ∈ {1.0, 0.85, 0.6}` | single cell (wrong) / `founder_size` inferred (`'law.founder_sizes=[1,2,3]'`) |

(These overrides compose and build: checked with `build_components` on 2026-11-05,
together with the 1c, 2b, 3a and 3b overrides.)

- Output: bias of each A and B parameter vs `f` / `founder_size.1`; PPC; whether the
  inferred founder rows are themselves recovered.
- Decision: if the dynamics are biased under a wrong founder but recovered when the
  founder is inferred, always infer the founder rows on real data. If the founder
  rows cannot be recovered, they must be measured (e.g. imaging at seeding).
- Crossed with GW and BH, 5 seeds: 2 × (4 + 3) × 2 × 5 = 140 runs.

### 1b – Per-type death → number of observed organoids

Question: how strongly do `A.death` and `B.death` change the fraction of seeded
organoids that are detected at day 6 and day 10? (Selection on detection is
informative, but the real data report only the organoids observed.)

- No inference needed. Grid over `A.death ∈ {0, .03, .1, .2, .3}` × `B.death ∈ {0, .1,
  .3, .5}` (the other entries of each row rescaled), GW (exact `s_G`) and BH
  (simulated), at horizons 6/10 days. Needs P2.
- Output: heat map of `detection_rate`. If the real experiment knows how many
  organoids were seeded, the observed fraction is an extra datum. Then also test
  adding it as a summary feature (`observation=all_clones`, which keeps empty clones),
  and its effect on the death parameters' contraction.

### 1c – Fix the death rates, infer the rest

Question: if `A.death` and `B.death` are fixed (to a wrong value) and everything
else is inferred, do other parameters compensate (a family of equally good solutions)?
If they do, death has to be measured independently (e.g. an apoptosis marker).

- Grid: `+law.fixed.A.death=dA +law.fixed.B.death=dB`, `dA ∈ {0, .03 (true), .1, .2}`,
  `dB ∈ {0, .1 (true), .2, .4}`; the data always use the truth. GW, 5 seeds:
  16 × 5 = 80 runs (+ BH for the corners).
- Output per `(dA, dB)`: PPC `fraction_outside_95`, the C2ST of
  `scripts/truth_vs_posterior.py`, and the inferred values of the other parameters.
- Reading: a **flat PPC/C2ST valley** across `(dA, dB)` means death is not
  identifiable (infinite solutions), and the compensating parameters (likely `persist`
  and the division split) show the direction of the ridge. A **sharp minimum** at the
  truth means the data constrain death, and the free-death runs should agree
  (compare the contraction of `A.death` and `B.death` in the baseline).

---

## E2. Typing

### 2a / 2b – Mis-typing between A and B

The clustering puts cells of a third population into A or B, so labels are wrong at
some rate. We separate two cases:

* **2a – known error (information loss)**: data and inference both use
  `observation=noisy` with `capture=1.0`, `confusion = e`.
* **2b – ignored error (bias)**: data `observation@data.observation=noisy
  data.observation.capture=1.0 data.observation.confusion=e`; inference `perfect`.

`e ∈ {0, 0.02, 0.05, 0.10, 0.15, 0.20}` (the biological estimates are ≈5% double
positive and ≈15% double negative). Asymmetric version: `confusion={A: {A: 1-e1, B: e1},
B: {A: e2, B: 1-e2}}`. GW and BH, 5 seeds: 6 × 2 × 2 × 5 = 120 runs.

- Output: error and contraction vs `e` for every parameter, and PPC (does the
  misspecified model reveal itself?).
- Decision: the largest tolerable `e` (bias below the posterior std) is the target for
  the clustering. If 2b is biased but 2a is not, the error rate must be estimated
  (e.g. 2D vs 3D comparison, Q3) and passed to the model.

### 2c – A third type

The double negative (and/or double positive) cells are a real type C that the markers
read as A or B. Needs P3.

| Run | Data | Inference |
|---|---|---|
| ignore C | `law@data.law=three_type observation@data.observation=three_split` | two-type `reversible_ab`, `perfect` |
| model C, free | same | `law=three_type observation=three_split` (rectangular confusion, assumed known) |
| model C, constrained | same | `law=three_type_traj observation=three_split` |

Sweep `q ∈ {0, 0.5, 1}` (C read as B, split, read as A) and the C fraction (through
the `A.to_C` / `A.div_AC` reference values, about 5% and 15% of cells).

- Output: recovery of the A and B rows in all three, contraction of the C row, PPC.
- Decision: whether adding C is worth it (does the PPC flag the two-type model? do
  the A/B estimates change?), and whether C is identifiable at all without its own
  marker, or only with the trajectory constraint.

Status: DONE. Original truth (`results/e2c/`) was in the three-type prior tail, so even
the true model failed. Rerun with 10 truths drawn from the prior
(`results/e2c_prior/`, `+experiment=e2c_prior_truth`): ignoring C is flagged at q = 0
(9/10) and q = 0.5 (7/10) but not at q = 1 (2/10, confident bias); modelling C is
calibrated but uninformative (A/B error = prior-mean baseline); the trajectory
constraint is violated by every prior truth and still undetected.

---

## E3. Measurement error: division times and desynchronisation (BH only)

The real division time is estimated globally, but it may differ by type, and cells
desynchronise (the lifetime CV).

- **3a – Per-type division time**: data with `A.mean_lifetime = 20`, `B.mean_lifetime ∈
  {20, 30, 45}`. Inference (i) infers both (baseline), (ii) fixes both to a common
  global estimate (`+simulator.model.fixed.A.mean_lifetime=T
  +simulator.model.fixed.B.mean_lifetime=T`, T = the true pooled mean), (iii) fixes them
  to a wrong value (±25%).
- **3b – Desynchronisation**: data `data.simulator.model.lifetime_shape ∈ {1, 2, 4, 8,
  32}` (CV 1 → 0.18); inference `lifetime_shape = 8`. Also per-type shapes
  (`{A: 4, B: 16}`).
- **3c – Inactivation / quiescence**: cells that stop cycling (outcome `persist`) trade
  off against longer lifetimes. Test whether `persist` and `mean_lifetime` are
  confounded: the posterior correlation in the baseline, and fixing `persist` to 0 when
  the data have `A.persist ∈ {0.03, 0.15, 0.3}`. *(To confirm with the biologists that
  "inattivazione" means quiescence; see Q4.)*
- 5 seeds each; about 3·3 + 5 + 2 + 3 = 19 settings, so about 95 runs.
- Output: bias in the outcome probabilities caused by a wrong time model, and
  whether the PPC flags it.
- Decision: whether a single global division time is adequate, and how precisely it
  has to be measured.

---

## E4. Amount of data (design benchmark)

Question: how many organoids, and which days, are needed to reach a target accuracy
for each parameter?

- `design.clones ∈ {16, 32, 64, 128, 256, 512}` × horizons {day 10}, {6, 10},
  {4, 6, 10}, {6, 10, 14} (GW generations or BH hours). Baseline law and observation,
  then repeated at the error level judged realistic in E2 (e.g. `confusion=0.05`).
  5 seeds: 6 × 4 × 5 = 120 runs per simulator.
- Output: for each parameter, contraction and error vs number of organoids; the
  **minimum organoids per day** to reach contraction > 0.5 and |error| < 0.05.
  Plot one curve per horizon set.
- Also: with `law=two_type` at 256–512 clones, does `B.to_A` become recoverable (E0's
  C2ST predicts the signal appears around 512 clones)?
- Deliverable: a table *"to estimate X to ±0.05 you need N organoids at days D"* for
  experiment planning.
- **First slice done** (`+experiment=design_comparison`, `results/design_comparison/info.md`):
  `design ∈ {single_128 ([12]×128), three_128 ([4,8,12]×128), single_384 ([12]×384)}`
  × `law ∈ {two_type, reversible_ab}` × 5 seeds, NPE 20k. `single_384` matches
  `three_128`'s total clones, separating time points from clone count. Result: 3
  endpoints beat 1 at equal total clones (MAE 0.028 vs 0.058, reversible_ab).
- **E4 BH done** (`results/e4/info.md`); E4 GW running (`results/e4/gw`).

---

## X. Cross-cutting checks (after E1–E4, on the configurations that matter)

- [ ] **X1 – Methods**: NPE vs NLE vs NRE (and sequential NPE) at the baseline and at
      the worst realistic error setting (`+experiment=compare_methods`). NLE/NRE with
      `budget.sbc_cases=50`.
      Proposed scope (2026-11-06; all results so far are amortised NPE): rerun the
      decisive settings with NPE / SNPE / NLE / NRE, 5 seeds each (~80 runs): E0
      B divisions free vs fixed, BH ridge reference truth, one silent misspecification
      (E2c q = 1 or E3a too-short lifetimes), one loud one (X5b wrong lifetime prior).
      Question: are identifiability findings method-independent, and does the
      flagged/silent pattern change (support mass is NPE-specific; MCMC methods fail
      with confident posteriors inside the prior)?
- [x] **X2 – Seeds** (`results/x2/`, 10 × 2 arms × GW/BH): training noise ≈ 20% of the
      posterior sd; training ≈ half (GW) to 70% (BH) of the seed-to-seed variance;
      quote posterior sd, not seed sd. The BH ridge bias is identical across datasets
      and training seeds.
- [ ] **X3 – Model**: GW vs BH on the same question. Can a GW fit describe BH data?
      GW ↔ hours mapping implemented (`simulator=galton_watson_hours`,
      `hours_per_generation` τ, h → round_half_up(h/τ) generations); preset
      `x3_gw_fit_bh` sweeps τ ∈ {16, 20, 24, 30} × 5 seeds, BH control arm in its header.
- [ ] **X4 – Combined errors**: the realistic combination (organoid death + `e` + C +
      lifetime CV) at the real design, `budget=full`, 5 seeds. This is the error budget
      to quote alongside the real-data fit.
- [x] **X5 – Role of the prior** (`results/x5/`): truth position × prior shape (X5a),
      informative lifetime prior right/wrong (X5b), organoids needed to overrule a
      tail prior (X5c), prior-averaged calibration GW/BH (X5d). Methods calibrated on
      average; the B-row failure is structural (constraints fix it, priors and more
      organoids do not); the BH ridge is a tail-truth effect, helped by a correct
      lifetime prior; a too-long lifetime prior is detected.

---

## R. Real data (CRC0322)

- [ ] Convert `submodules/GW/data/df_6D_withmapping.csv` (tab-separated, 165
      organoids) and `df_10D_withmapping_new.csv` (101) into the `horizon,A,B` format
      (A = `n_LGR5`, B = `n_KRT20`). Decide between the 2D counts (`n_*`) and the 3D
      estimates (`*_sample1..3`) (Q3).
- [ ] Train with the final model from E1–E4 at the real design (`budget=full`), check
      SBC, fit with `bio-sbi-infer`, and run the PPC on the real data.

---

## Q. Open biological questions (block E2c / E3c / R)

- [ ] **Q1** – Does the third type only go towards differentiated cells (trajectory
      inference)? This sets the constraints in `three_type_traj`.
- [ ] **Q2** – Proportions: about 5% double positive (++), about 15% double negative (−−).
      Per day? Per organoid?
- [ ] **Q3** – How much mislabelling is there from 2D to 3D? This gives the `e` to use
      in E2 and decides which counts to fit in R.
- [ ] **Q4** – What does "inattivazione" (inactivation) mean biologically:
      quiescence, or a drug/genetic inactivation acting on division time?
- [ ] **Q5** – Is the number of seeded organoids known (for 1b)? Is organoid death
      known to be type-independent?

---

## Suggested order

1. P1, P2, P0 (small code changes) → finish E0.
2. E4 baseline (sets the realistic design for everything else) and E1c (decides
   whether death must be fixed).
3. E2a/2b and E1a in parallel; E3 after P0.
4. E2c once Q1/Q2 are answered (needs P3).
5. X1–X4, then R.
