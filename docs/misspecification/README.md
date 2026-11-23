# Misspecification study: exact-likelihood two-type GW on organoid clone counts

**Question.** The exact GW analysis assumes: every clone starts from **one LGR5⁺ (A) cell**,
cell types are **scored without error**, every clone follows the **same** outcome
probabilities, divisions are **synchronous** generations, rates do **not depend on clone
size or time**, and **every cell is counted**. What happens to the estimates (especially
B→A, dedifferentiation), the intervals and the dedifferentiation test when the data come
from biology or an assay that breaks one of these assumptions? Can the problem be seen in
the data, and does modelling it fix it?

**Status (2026-11-23 02:15): PRELIMINARY.** MLE, profile intervals and goodness of fit for
datasets 0–4 of every scenario × truth are complete (`tables/scenario_table.md`,
`tables/per_scenario_truth.md`, `figures/bias_heatmap.png`, `figures/coverage_toA_heatmap.png`).
The extended-model fits, the dedifferentiation-test false-positive runs, NUTS and datasets 5–9
were still running from `scripts/misspecification/queue.sh` (restartable). Refresh the tables
and figures with `uv run python scripts/misspecification/analyse.py` when the queue ends.
Design and methods: `design.md.part` (to be merged here).

## First findings (standard model, datasets 0–4, n = 512)
- **Mis-typing barely affects the estimates** (B→A, m_BA, ρ, stable fraction ≈ base; 95%
  profile coverage of B→A 0.78–0.96 vs 0.78 on base; G² goodness of fit cannot detect it:
  4–9% vs 4% false alarms). Its danger is the dedifferentiation test (see docs/dediff_test).
- **Founder problems, fate heterogeneity and asynchrony push B→A down and B's division mass
  up** (the B→A ↔ B→AA/AB "parent swap" weak direction): for T1/T3/T4 B→A is estimated
  0.35–0.8 too low, B.div 0.5–1 too high; m_BA and ρ move much less (m_BA +0.1–0.3, ρ ≤ 5%,
  HET10 up to +14%). B→A 95% coverage 0.07–0.26 (D15, ASYNC, HET), 0.63 (F70).
- **Doublets are worse than founder-type mixing:** with founder type 70/30 + 15% doublets,
  allowing doublets in the model is what restores the fit (≈ 400 nats on T1 seed 0); the
  founder fraction itself is weakly identified (flat over 0.5–1 within a few nats).
- **Density dependence and slowing growth** bias ρ by −5 to −9% and m_BA by up to −0.7 (P
  truths); detected by G² in 76–100% of datasets. HET10/HET50 detected 80–100%, ASYNC 82%,
  doublets 51–58%, founder-type mixture alone 9%, cell loss 4–9%.
- **Cell loss** (10–20%) inflates B→A RMSE ×1.8–2.2 and is essentially undetectable by G².
- Knowing the true setting (founders / confusion / capture) restores coverage to 0.78–0.93
  (base 0.78).
