"""Death-free dataset D0: Fisher information and weak directions.

Three settings per truth (XG2 design: A founders, one clone at each of generations 4/6/8/10
per unit n; prior whitening by 42 as in the weak-direction study):

* ``orig``          original truth, standard model (10 tangent directions);
* ``D0 free``       death-free truth, death estimated (10 directions; the death directions
                    are one-sided at the boundary, Fisher = local curvature);
* ``D0 nodeath``    death-free truth, death fixed at 0 (8 directions on the face).

Reported per truth and setting: weakest prior-whitened eigenvalue (1/lam_min clones per
endpoint) and its eigenvector; sd of B->A and total variance of the 12 probabilities at
n = 512 clones per endpoint; for the 5 parent swaps, 5 average-cell changes and some B-row
contrasts, n_half = clones per endpoint at which the posterior variance along the direction
falls to half the prior's (all other directions estimated).

  uv run python scripts/death_free/fisher.py

Info cache: ``results/exact_gw/weak_directions/info_D0_<truth>.npz``; tables
``docs/death_free/tables/fisher_*.csv``; figure ``docs/death_free/figures/fisher_d0.png``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "weak_directions"))
import analyse_fisher as WA  # noqa: E402
import common as W  # noqa: E402  (weak_directions.common)
import dfcommon as D  # noqa: E402

N_EP = 512  # clones per endpoint
FULL = W.row_contrasts()


def tied_basis() -> np.ndarray:
    """Orthonormal basis (12, 5) of the mirror-tied directions (A change c, B change = c
    moved to the mirrored entries)."""
    cols = []
    for c in np.linalg.qr(np.eye(6) - 1 / 6)[0][:, :5].T:
        v = np.zeros(12)
        v[:6] = c
        v[6 + np.array(D.MIRROR_B_OF_A)] = c
        cols.append(v / np.linalg.norm(v))
    return np.array(cols).T


BASIS = {"free": FULL,
         "nodeath": W.row_contrasts(np.array([i not in D.I_DEATH for i in range(12)])),
         "noswitch": W.row_contrasts(np.array([i not in D.I_SWITCH for i in range(12)])),
         "tied": tied_basis()}  # fmt: skip
LABEL = {"nodeath": "death fixed 0", "noswitch": "switch fixed 0", "tied": "B tied to A"}
TITLE = {"D0": "no death (D0)", "S0": "no switching (S0)", "M": "mirror roles (M)"}


def settings(kind: str) -> dict[str, tuple[str, np.ndarray]]:
    """Setting label -> (truth kind, basis): original, kind free, kind constrained."""
    con = D.CONSTRAINED[kind]
    return {"orig": ("orig", FULL), f"{kind} free": (kind, FULL),
            f"{kind} {con}": (kind, BASIS[con])}  # fmt: skip


def info_K(kind: str, gt: str) -> np.ndarray:
    """Whitened information (12 x 12) of one A-founded clone at each of 4/6/8/10."""
    if kind == "orig":
        info = WA.load_info(gt)
    else:
        info = W.info_cache(f"{kind}_{gt}", D.truth(gt, kind).ravel(), gens=W.HORIZONS,
                            founders=("A",))  # fmt: skip
    return WA.design_info(info, WA.BASE) / W.PRIOR_PREC


def n_half(K: np.ndarray, B: np.ndarray, d: np.ndarray, frac: float = 0.5) -> float:
    """Clones per endpoint with posterior variance along unit d <= frac x prior (basis B)."""
    Kc, dc = B.T @ K @ B, B.T @ d
    if np.linalg.norm(dc) < 1e-9:
        return np.nan
    dc = dc / np.linalg.norm(dc)
    k = B.shape[1]

    def ratio(n):
        return dc @ np.linalg.solve(np.eye(k) + n * Kc, dc)

    lo, hi = 1e-3, 1e9
    if ratio(hi) > frac:
        return np.inf
    for _ in range(100):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if ratio(mid) > frac else (lo, mid)
    return float(hi)


def directions(p12: np.ndarray) -> dict[str, np.ndarray]:
    _, u = WA.stable_composition(p12)
    names, nb = WA.named_basis(u)
    out = {n.split(":")[0]: nb[:, i] for i, n in enumerate(names)}
    ix = {n: i for i, n in enumerate(D.NAMES)}

    def vec(**kw):
        v = np.zeros(12)
        for k, x in kw.items():
            v[ix[k.replace("__", ".")]] = x
        return WA.tangent(v)

    out.update({
        "B to_A vs B death": vec(B__to_A=1, B__death=-1),
        "B to_A vs B persist": vec(B__to_A=1, B__persist=-1),
        "B to_A vs B div_AB": vec(B__to_A=1, B__div_AB=-1),
        "B divides vs not": vec(B__div_AA=1, B__div_AB=1, B__div_BB=1, B__death=-1,
                                B__to_A=-1, B__persist=-1),
    })  # fmt: skip
    return out


def analyse(gt: str, setting: str, sets: dict) -> tuple[dict, dict]:
    kind, B = sets[setting]
    p12 = D.truth(gt, kind).ravel()
    K = info_K(kind, gt)
    lam, V = W.eig_tangent(K, B)
    Kb = B.T @ (K * W.PRIOR_PREC) @ B  # unwhitened, per unit n
    cov = B @ np.linalg.inv(N_EP * Kb) @ B.T
    e = np.zeros(12)
    e[D.NAMES.index("B.to_A")] = 1.0
    row = {"truth": D.SHORT[gt], "setting": setting, "dim": B.shape[1],
           "lam_min": float(lam[0]), "inv_lam_min": float(1 / lam[0]),
           "weakest": W.label(V[:, 0]), "second": W.label(V[:, 1]),
           "inv_lam_2": float(1 / lam[1]),
           "sd_BtoA": float(np.sqrt(e @ cov @ e)), "trvar12": float(np.trace(cov)),
           "n_weak_le_512": int(np.sum(1 / lam > N_EP))}  # fmt: skip
    nh = {"truth": D.SHORT[gt], "setting": setting}
    for name, d in directions(p12).items():
        # constrained models: a direction mostly outside the model (>= half its squared
        # norm) does not exist; others are projected on the model's directions
        inside = float(np.sum((B.T @ d) ** 2))
        nh[name] = np.nan if inside <= 0.5 + 1e-9 and B is not FULL else n_half(K, B, d)
    return row, {**nh, "_lam": lam}


def figure(kind: str, tab: pd.DataFrame, nh: pd.DataFrame, spectra: dict) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: PLC0415

    SETTINGS = settings(kind)
    cols = dict(zip(SETTINGS, ["#8a8984", "#2a78d6", "#eb6834"], strict=True))
    truths = [D.SHORT[g] for g in D.gts(kind)]
    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25])
    # (a) 1/lambda spectra
    ax = fig.add_subplot(gs[0, :2])
    w = 0.27
    for k, s in enumerate(SETTINGS):
        for i, t in enumerate(truths):
            inv = np.sort(1 / spectra[(t, s)])[::-1]
            xs = np.full(len(inv), i + (k - 1) * w)
            ax.scatter(xs, inv, s=14, color=cols[s], label=s if i == 0 else None, zorder=3)
    ax.axhline(N_EP, color="0.4", lw=0.8, ls="--")
    ax.text(len(truths) - 0.5, N_EP * 1.15, "512 clones / endpoint", ha="right", fontsize=8,
            color="0.3")  # fmt: skip
    ax.set_yscale("log")
    ax.set_xticks(range(len(truths)), truths)
    ax.set_ylabel("1/λ: clones per endpoint where data = prior")
    ax.set_title("(a) All directions (eigenvalues of the prior-whitened information); "
                 "above the line = weaker than the prior at n = 512", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", color="0.92")
    # (b) B->A sd
    ax = fig.add_subplot(gs[0, 2])
    for k, s in enumerate(SETTINGS):
        v = tab[tab.setting == s].set_index("truth").loc[truths, "sd_BtoA"]
        ax.barh(np.arange(len(truths)) + (k - 1) * w, v, height=w, color=cols[s], label=s)
    ax.set_yticks(range(len(truths)), truths)
    ax.invert_yaxis()
    ax.set_xlabel("asymptotic sd of B→A at n = 512 / endpoint")
    ax.grid(axis="x", color="0.92")
    ax.set_title("(b) B→A precision", fontsize=10)
    # (c) n_half heatmaps
    dcols = [c for c in nh.columns if c not in ("truth", "setting", "_lam")]
    cmap = LinearSegmentedColormap.from_list("b", ["#f7fafe", "#9ec5f4", "#3987e5", "#184f95",
                                                   "#0d366b"])  # fmt: skip
    cmap.set_bad("#e9e8e4")
    for k, s in enumerate(SETTINGS):
        ax = fig.add_subplot(gs[1, k])
        M = nh[nh.setting == s].set_index("truth").loc[truths, dcols].to_numpy(float)
        Mc = np.where(np.isfinite(M), np.clip(M, 1, 1e6), np.nan)
        ax.imshow(np.ma.masked_invalid(Mc), cmap=cmap, norm=LogNorm(1, 1e6), aspect="auto")
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                v = M[a, b]
                txt = ("—" if np.isnan(v) else "∞" if np.isinf(v) else "<1" if v < 1 else
                       f"{v:.0f}" if v < 1e4 else f"{v:.0e}".replace("e+0", "e"))  # fmt: skip
                dark = np.isfinite(v) and v > 3e3
                ax.text(b, a, txt, ha="center", va="center", fontsize=6.5,
                        color="white" if dark else "#1f1f1d")  # fmt: skip
        ax.set_xticks(range(len(dcols)), dcols, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(truths)), truths if k == 0 else [""] * len(truths), fontsize=8)
        ax.tick_params(length=0)
        ax.set_title(f"(c) {s}: clones / endpoint to halve the prior variance", fontsize=9)
    con = LABEL[D.CONSTRAINED[kind]]
    fig.suptitle(f"Structured truths, {TITLE[kind]}: information on the weak directions, "
                 f"standard model (blue) vs {con} (orange); grey: original truth", fontsize=11)
    out = D.DOCS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"fisher_{kind}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default=",".join(D.KINDS))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    for kind in ap.parse_args().kinds.split(","):
        sets = settings(kind)
        rows, nhs, spectra = [], [], {}
        for gt in D.gts(kind):
            t0 = time.time()
            for s in sets:
                r, nh = analyse(gt, s, sets)
                rows.append(r)
                spectra[(r["truth"], s)] = nh.pop("_lam")
                nhs.append(nh)
            print(f"{kind} {gt}: {time.time() - t0:.0f} s", flush=True)
        tab, nh = pd.DataFrame(rows), pd.DataFrame(nhs)
        tdir = D.DOCS / "tables"
        tdir.mkdir(parents=True, exist_ok=True)
        tab.to_csv(tdir / f"fisher_summary_{kind}.csv", index=False)
        nh.to_csv(tdir / f"fisher_n_half_{kind}.csv", index=False)
        figure(kind, tab, nh, spectra)
        print(tab.round(4).to_string())
        print(nh.round(0).to_string())


if __name__ == "__main__":
    main()
