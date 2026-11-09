# B-row identifiability: exact GW analysis

Code behind `docs/b_row_identifiability/note.pdf`. Galton–Watson, two types, one A
founder, detected clones (perfect capture), truth θ* = the `two_type` reference law.
The law of Z_n is exact (2-D FFT of the pgf on roots of unity, forward-mode derivatives),
not simulated.

| Script | Produces (note section) | Time |
|---|---|---|
| `gwexact.py` | library: exact law, derivatives, conditioned Fisher information, KL | — |
| `fisher.py` then `fisher2.py` | Tables 1–2: Fisher information per endpoint, Cramér–Rao bounds per design (writes `fisher*.npz` in the cwd) | ~1 min |
| `varW.py` | Theorem 3 check (Var W formula vs exact law) and a compensated pair | seconds |
| `llr.py` | §4 item 2: exact log-LR of θ* vs NPE posterior means on the E0-at-4096 reference data. Needs `results/e4/gw_xxl_two_type_*`; set `BIO_SBI_RESULTS=/path/to/results` | ~1 min |
| `summary_twopoint.py` | §4 item 3: can the `moments` summary separate θ* from the NPE answer? | ~5 min |
| `profile.py [workers]` | §5: profile KL over the B-division mass, writes `profile.jsonl` (kept in git) | ~25 min with 3 workers |

Run from the repository root: `uv run python scripts/b_row_theory/<script>.py`.
