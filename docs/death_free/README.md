# Structured truths: no death (D0), no switching (S0), mirror roles (M)

Report section 13 (`report_exact_gw/report.pdf`). How the weak directions and inference change
when the truth has structure, fitted with the standard 12-probability model and with the true
structure built in.

## Datasets (from the 9 XG2 truths, `scripts/death_free/dfcommon.py`)

| kind | truth | constrained model |
|---|---|---|
| D0 | A.death = B.death = 0, rows rescaled (no extinction: every clone detected) | `nodeath` (death fixed 0, 8 directions) |
| S0 | A.to_B = B.to_A = 0, rows rescaled (T1/T2/T4: B terminal) | `noswitch` (8 directions) |
| M | B row = mirror of A row (labels swapped); T1-T3 coincide -> 7 truths | `tied` (5 directions) |

Data: misspecification-study simulator, 512 detected clones per generation 4/6/8/10, A
founders, rng `[truth index, seed, 0]` (paired with the original truths' base datasets 0-9).

## Scripts

- `fisher.py` - prior-whitened Fisher information (weak-direction study code), settings
  original / standard model / constrained model; 1/lambda spectra, B->A sd and all-12 total
  variance at n = 512, n_half along the parent swaps, average-cell changes and B-row
  contrasts. Tables `tables/fisher_{summary,n_half}_<kind>.csv`, figures `figures/fisher_<kind>.png`.
- `run.py` - multistart MLE (datasets 0-9, both models) and NUTS (dataset 0; the tied model
  has no NUTS: the law cannot tie rows). Results `results/exact_gw/death_free/{mle,nuts}/`.
- `queue.sh` (D0), `queue_s0_m.sh` (S0, M) - local job queues.
- `analyse.py` - MLE / NUTS summaries against the original truths (misspecification base fits);
  tables `tables/{mle_runs,mle_summary,nuts_runs}_<kind>.csv`, figures `figures/sim_<kind>.png`.

## Main results (MLE, 10 datasets; Fisher at 512 clones per endpoint)

- D0: death found ~0 by the standard model (exactly 0 in 20-80% of fits); the "B switches back
  vs B dies" ridge disappears; T1-T4 B->A RMSE 0.07-0.27 -> 0.04-0.07 (0.03-0.05 death fixed);
  all-12 error halves; P truths gain less (faster growth weakens the parent swaps) but all
  improve in simulation.
- S0: where B never divides, B is terminal and everything is easy (weakest direction 20-54
  clones per endpoint); where B divides the parent swaps get harder (P1 1.8k -> 8.2k).
- M: hardest case for the standard model (P1 weakest direction 57k clones per endpoint, 70% of
  B->A fits on the far branch; all-12 error larger than for the original truths); tying B to A
  removes the mixed-division, death and division swaps and halves the all-12 error.
- NUTS: dataset 0 per setting; T2 (D0, death estimated) stuck on the death branch (R-hat
  1.38); several S0/M runs mixed poorly (R-hat up to 1.7). Truths at 0 (deaths in D0, B->A in
  S0) can never be inside a central 90% interval, which lowers the reported coverage.
