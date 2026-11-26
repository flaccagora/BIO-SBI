"""Death-free dataset D0: analysis of the MLE and NUTS runs.

Settings per truth: ``orig`` (original truth, standard model: base fits of the
misspecification study on the paired datasets 0-9), ``D0 free`` (death estimated), ``D0 nodeath``
(death fixed at 0). MLE: B->A RMSE and bias, share of fits with |B->A error| > 0.2 (far branch),
mean |error| over the 12 probabilities, the fitted deaths (D0 free), errors of rho / stable A
fraction. NUTS (dataset 0): |posterior mean - truth|, sd and 90% coverage of B->A, all-12
error and coverage, posterior of the deaths.

  uv run python scripts/death_free/analyse.py

Writes ``docs/death_free/tables/{mle_runs,mle_summary,nuts_runs}.csv`` and
``docs/death_free/figures/sim_d0.png``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dfcommon as D  # noqa: E402

C = D.C
KIND = "D0"  # set by main


def settings(kind: str) -> list[str]:
    return ["orig", f"{kind} free", f"{kind} {D.CONSTRAINED[kind]}"]


SETTINGS = settings(KIND)
MISSP = Path("/home/matteo/BIO-SBI/results/exact_gw/misspecification/fits/base")


def _row(gt, seed, setting, P, truth, extra=None):
    e = {k: P[k] - truth[k] for k in D.NAMES}
    sP, sT = C.summaries(np.array([P[k] for k in D.NAMES]).reshape(2, 6)), \
        C.summaries(np.array([truth[k] for k in D.NAMES]).reshape(2, 6))
    return {"truth": D.SHORT[gt], "seed": seed, "setting": setting, "err_BtoA": e["B.to_A"],
            "mae12": float(np.mean([abs(v) for v in e.values()])),
            "A.death": P["A.death"], "B.death": P["B.death"],
            "err_rho": sP["rho"] - sT["rho"], "err_stableA": sP["stableA"] - sT["stableA"],
            **(extra or {})}  # fmt: skip


def mle_runs(kind: str) -> pd.DataFrame:
    rows = []
    for gt in D.gts(kind):
        for seed in range(10):
            f = MISSP / gt / f"s{seed}.json"
            if f.exists():
                d = json.loads(f.read_text())
                rows.append(_row(gt, seed, "orig", d["std"]["P"], d["truth"]))
            for model in ("free", D.CONSTRAINED[kind]):
                f = D.OUT / "mle" / f"{kind}_{model}" / f"{gt}_s{seed}.json"
                if f.exists():
                    d = json.loads(f.read_text())
                    rows.append(_row(gt, seed, f"{kind} {model}", d["P"], d["truth"],
                                     {"lr_vs_truth": 2 * (d["ll"] - d["ll_truth"])}))
    return pd.DataFrame(rows)


def mle_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["truth", "setting"])
    s = pd.DataFrame({
        "n": g.size(),
        "rmse_BtoA": g.err_BtoA.apply(lambda v: float(np.sqrt(np.mean(v ** 2)))),
        "bias_BtoA": g.err_BtoA.mean(),
        "far_branch": g.err_BtoA.apply(lambda v: float(np.mean(np.abs(v) > 0.2))),
        "mae12": g.mae12.mean(),
        "mean_A_death": g["A.death"].mean(), "mean_B_death": g["B.death"].mean(),
        "share_death_0": g.apply(lambda d: float(np.mean((d["A.death"] < 1e-3)
                                                         & (d["B.death"] < 1e-3)))),
        "rmse_rho": g.err_rho.apply(lambda v: float(np.sqrt(np.mean(v ** 2)))),
        "rmse_stableA": g.err_stableA.apply(lambda v: float(np.sqrt(np.mean(v ** 2)))),
    }).reset_index()  # fmt: skip
    order = [D.SHORT[g] for g in D.GTS]
    s["truth"] = pd.Categorical(s.truth, order, ordered=True)
    s["setting"] = pd.Categorical(s.setting, SETTINGS, ordered=True)
    return s.sort_values(["truth", "setting"])


def nuts_runs(kind: str) -> pd.DataFrame:
    rows = []
    lab = {"orig_free": "orig", f"{kind}_free": f"{kind} free",
           f"{kind}_{D.CONSTRAINED[kind]}": f"{kind} {D.CONSTRAINED[kind]}"}
    keep = {D.SHORT[g] for g in D.gts(kind)}
    for f in sorted((D.OUT / "nuts").glob("*/*.json")):
        if f.parent.name not in lab:
            continue
        d = json.loads(f.read_text())
        if D.SHORT[d["gt"]] not in keep:
            continue
        st = d["stats"]
        b = st["B.to_A"]
        rows.append({"truth": D.SHORT[d["gt"]], "setting": lab[f"{d['kind']}_{d['model']}"],
                     "BtoA_abs_err": abs(b["mean"] - b["truth"]), "BtoA_sd": b["sd"],
                     "BtoA_cover90": b["cover90"],
                     "mae12": float(np.mean([abs(st[p]["mean"] - st[p]["truth"]) for p in D.NAMES])),
                     "cover12": float(np.mean([st[p]["cover90"] for p in D.NAMES])),
                     "A_death_mean": st["A.death"]["mean"], "A_death_q95": st["A.death"]["q95"],
                     "B_death_mean": st["B.death"]["mean"], "B_death_q95": st["B.death"]["q95"],
                     "rhat": d["convergence"]["max_rank_rhat"],
                     "ess": d["convergence"]["min_ess_bulk"],
                     "stop": d["stop"]["reason"], "hours": d["seconds"] / 3600})  # fmt: skip
    return pd.DataFrame(rows)


RAMP = ["#f7fafe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def figure(s: pd.DataFrame, nt: pd.DataFrame) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm  # noqa: PLC0415

    cmap = LinearSegmentedColormap.from_list("seq", RAMP)
    cmap.set_bad("#e9e8e4")
    truths = [D.SHORT[g] for g in D.gts(KIND)]
    con = {"nodeath": "death\nfixed 0", "noswitch": "switch\nfixed 0", "tied": "B tied\nto A"}
    xl = ["original", f"{KIND}, standard\nmodel", f"{KIND}, {con[D.CONSTRAINED[KIND]]}"]
    panels = [("rmse_BtoA", "B→A RMSE (MLE, 10 datasets)", s, "{:.2f}"),
              ("mae12", "all 12: mean |MLE − truth|", s, "{:.3f}"),
              ("far_branch", "share of fits with |B→A error| > 0.2", s, "{:.0%}"),
              ("BtoA_abs_err", "B→A |posterior mean − truth|\n(NUTS, dataset 0)", nt, "{:.2f}"),
              ("mae12", "all 12: |posterior mean − truth|\n(NUTS, dataset 0)", nt, "{:.3f}")]
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 5.2), constrained_layout=True)
    for ax, (col, title, df, fmt) in zip(axes, panels, strict=True):
        M = np.full((len(truths), 3), np.nan)
        star = np.zeros_like(M, bool)
        for i, t in enumerate(truths):
            for j, st in enumerate(SETTINGS):
                if df.empty:
                    continue
                r = df[(df.truth == t) & (df.setting == st)]
                if len(r):
                    M[i, j] = float(r[col].iloc[0])
                    if df is nt and col == "BtoA_abs_err":
                        star[i, j] = not bool(r.BtoA_cover90.iloc[0])
        vmax = np.nanmax(M) if np.isfinite(M).any() else 1
        norm = PowerNorm(0.6, vmin=0, vmax=vmax)
        ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(3):
                v = M[i, j]
                txt = "—" if not np.isfinite(v) else fmt.format(v) + ("✱" if star[i, j] else "")
                dark = np.isfinite(v) and norm(v) > 0.55
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if dark else "#1f1f1d")  # fmt: skip
        ax.set_xticks(range(3), xl, fontsize=8)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(truths)), truths, fontsize=9)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=9.5, pad=6)
    title = {"D0": "no death (D0)", "S0": "no switching (S0)", "M": "mirror roles (M)"}[KIND]
    fig.suptitle(f"Structured truths, {title}, vs the original truths, n = 512 per endpoint "
                 "(lighter = smaller; ✱ = 90% interval of B→A misses the truth; — = not run)",
                 fontsize=11)  # fmt: skip
    out = D.DOCS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"sim_{KIND}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    global KIND, SETTINGS
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default=",".join(D.KINDS))
    tdir = D.DOCS / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    for kind in ap.parse_args().kinds.split(","):
        KIND, SETTINGS = kind, settings(kind)
        df = mle_runs(kind)
        df.to_csv(tdir / f"mle_runs_{kind}.csv", index=False)
        s = mle_summary(df)
        s.to_csv(tdir / f"mle_summary_{kind}.csv", index=False)
        nt = nuts_runs(kind)
        nt.to_csv(tdir / f"nuts_runs_{kind}.csv", index=False)
        figure(s, nt)
        print(kind)
        print(s.round(4).to_string(index=False))
        if len(nt):
            print(nt.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
