"""Lab-knowledge experiment, simulation side: MAP (6 datasets) and NUTS checks.

Compares, per truth (T1, T3, P3) and seeding, the founder models of ``run.py``:
``est`` (flat prior; plain MLE), ``lab20`` / ``lab100`` (informative prior, mode at the truth;
MAP), ``labw100`` (confident prior with p_A centred 0.15 too low; MAP) and ``known``.
Reported: B->A RMSE, mean absolute error over the 12 probabilities, |error| of p_A and of the
mean founder size, share of fits on the far branch of the B->A ridge (|error| > 0.2); NUTS:
B->A mean / sd / 90% coverage, all-12 error and coverage, p_A.

  uv run python scripts/founder_distribution/lab_prior_sim.py

Writes ``docs/founder_distribution/tables/lab_prior_{map,nuts}.csv`` and ``..._map_summary.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = Path("/home/matteo/BIO-SBI/results/exact_gw/founder_distribution")
TAB = ROOT / "docs" / "founder_distribution" / "tables"
NAMES = ["A.death", "A.persist", "A.to_B", "A.div_AA", "A.div_AB", "A.div_BB",
         "B.death", "B.to_A", "B.persist", "B.div_AA", "B.div_AB", "B.div_BB"]  # fmt: skip
MODELS = ["est", "lab20", "lab100", "labw100", "known"]
DISTS = ["S1_pA1.0", "S2_pA0.7", "S3_pA0.9", "S4_pA0.5"]
TRUTHS = ["T1_ref", "T3_Bdiv", "P3"]


def map_table() -> pd.DataFrame:
    rows = []
    for f in sorted((RES / "mle").glob("*/*/*.json")):
        d = json.loads(f.read_text())
        if d["model"] not in MODELS or d["gt"] not in TRUTHS:
            continue
        e = d["err"]
        rows.append({"truth": d["gt"], "dist": d["dist"], "model": d["model"], "seed": d["seed"],
                     "err_BtoA": e["B.to_A"], "mae12": float(np.mean([abs(e[p]) for p in NAMES])),
                     "abs_err_pA": abs(e.get("founder.A", 0.0)),
                     "abs_err_size": abs(e.get("mean_size", 0.0))})  # fmt: skip
    return pd.DataFrame(rows)


def summarise_map(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["truth", "dist", "model"])
    s = pd.DataFrame({
        "n": g.size(),
        "rmse_BtoA": g.err_BtoA.apply(lambda v: float(np.sqrt(np.mean(np.square(v))))),
        "bias_BtoA": g.err_BtoA.mean(),
        "far_branch": g.err_BtoA.apply(lambda v: float(np.mean(np.abs(v) > 0.2))),
        "mae12": g.mae12.mean(), "abs_err_pA": g.abs_err_pA.mean(),
        "abs_err_size": g.abs_err_size.mean()}).reset_index()  # fmt: skip
    s["model"] = pd.Categorical(s.model, MODELS, ordered=True)
    s["dist"] = pd.Categorical(s.dist, DISTS, ordered=True)
    s["truth"] = pd.Categorical(s.truth, TRUTHS, ordered=True)
    return s.sort_values(["truth", "dist", "model"])


def nuts_table() -> pd.DataFrame:
    rows = []
    for f in sorted((RES / "nuts").glob("*/*/*.json")):
        d = json.loads(f.read_text())
        if d["model"] not in MODELS:
            continue
        st = d["stats"]
        b = st["B.to_A"]
        rows.append({"truth": d["gt"], "dist": d["dist"], "model": d["model"],
                     "BtoA_mean": b["mean"], "BtoA_sd": b["sd"], "BtoA_truth": b["truth"],
                     "BtoA_cover90": b["cover90"],
                     "mae12": float(np.mean([abs(st[p]["mean"] - st[p]["truth"]) for p in NAMES])),
                     "cover12": float(np.mean([st[p]["cover90"] for p in NAMES])),
                     "pA_mean": st["founder.A"]["mean"], "pA_sd": st["founder.A"]["sd"],
                     "pA_truth": st["founder.A"]["truth"],
                     "rhat": d["convergence"]["max_rank_rhat"],
                     "ess": d["convergence"]["min_ess_bulk"],
                     "hours": d["seconds"] / 3600})  # fmt: skip
    return pd.DataFrame(rows)


SEEDLAB = {"S1_pA1.0": r"single, $p_A$ 1.0", "S2_pA0.7": r"mostly single, $p_A$ 0.7",
           "S3_pA0.9": r"1--3 cells, $p_A$ 0.9", "S4_pA0.5": r"1--5 cells, $p_A$ 0.5"}  # fmt: skip
TLAB = {"T1_ref": "T1", "T3_Bdiv": "T3", "P3": "P3"}
REPORT_TAB = ROOT / "report_exact_gw" / "tables"


def _cell(v: float | None, fmt: str) -> str:
    return "---" if v is None or not np.isfinite(v) else fmt.format(v)


def latex_map(s: pd.DataFrame) -> str:
    """Rows of the point-estimate table: B->A RMSE and all-12 MAE for the five models."""
    out = []
    for t in TRUTHS:
        for k, dist in enumerate(DISTS):
            r = s[(s.truth == t) & (s.dist == dist)].set_index("model")
            get = lambda m, v: float(r.loc[m, v]) if m in r.index else None  # noqa: E731
            cells = [TLAB[t] if k == 0 else "", SEEDLAB[dist]]
            cells += [_cell(get(m, "rmse_BtoA"), "{:.2f}") for m in MODELS]
            cells += [_cell(get(m, "mae12"), "{:.3f}") for m in MODELS]
            out.append(" & ".join(cells) + r" \\")
    return "\n".join(out) + "\n"


def latex_nuts(nt: pd.DataFrame) -> str:
    """Rows of the NUTS table: |posterior mean - truth| of B->A and all-12 MAE per model."""
    models = ["est", "lab100", "labw100", "known"]
    out = []
    keys = nt[["truth", "dist"]].drop_duplicates()
    keys = keys.assign(t=pd.Categorical(keys.truth, TRUTHS, ordered=True),
                       d=pd.Categorical(keys.dist, DISTS, ordered=True)).sort_values(["t", "d"])
    last = None
    for t, dist in zip(keys.truth, keys.dist, strict=True):
        r = nt[(nt.truth == t) & (nt.dist == dist)].set_index("model")
        get = lambda m, v: float(r.loc[m, v]) if m in r.index else None  # noqa: E731
        err = {m: (abs(get(m, "BtoA_mean") - get(m, "BtoA_truth")) if m in r.index else None)
               for m in models}  # fmt: skip
        cov = {m: ("" if m not in r.index else ("" if r.loc[m, "BtoA_cover90"] else "$^*$"))
               for m in models}  # fmt: skip
        cells = [TLAB[t] if t != last else "", SEEDLAB[dist]]
        cells += [_cell(err[m], "{:.2f}") + cov[m] for m in models]
        cells += [_cell(get(m, "mae12"), "{:.3f}") for m in models]
        out.append(" & ".join(cells) + r" \\")
        last = t
    return "\n".join(out) + "\n"


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    df = map_table()
    df.to_csv(TAB / "lab_prior_map.csv", index=False)
    s = summarise_map(df)
    (TAB / "lab_prior_map_summary.md").write_text(s.round(3).to_markdown(index=False))
    pd.set_option("display.width", 220)
    for v in ("rmse_BtoA", "mae12", "far_branch", "abs_err_pA", "abs_err_size"):
        print(v)
        print(s.pivot_table(index=["truth", "dist"], columns="model", values=v, observed=True)
              .round(3).to_string())  # fmt: skip
    print("n per cell:", s.n.min(), "-", s.n.max())
    nt = nuts_table()
    nt.to_csv(TAB / "lab_prior_nuts.csv", index=False)
    print(nt.round(3).to_string())
    REPORT_TAB.mkdir(parents=True, exist_ok=True)
    (REPORT_TAB / "founder_sim_point.tex").write_text(latex_map(s))
    (REPORT_TAB / "founder_sim_nuts.tex").write_text(latex_nuts(nt))


if __name__ == "__main__":
    main()
