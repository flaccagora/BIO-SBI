"""Eigen-analysis of the exact expected Fisher information per XG2 truth (weak directions).

Reads the cached per-generation information (``compute_info.py``) and writes the tables
and figures (a)-(c) of ``docs/weak_directions/``:

* spectra (prior-whitened eigenvalues lam; 1/lam = clones per endpoint at which the data
  match the flat prior along the eigenvector), weakest eigenvectors with loadings,
  interpretation diagnostics (mean-matrix share, parent-swap share, multinomial scale),
* recurrence across truths (subspace overlaps), boundary variants (full / face / eps),
* generation shares and counterfactual designs (early / late endpoints, B founders),
* named reference directions (Section 6 ridge, which-parent, ...) and B.to_A's variance.

  uv run python scripts/weak_directions/analyse_fisher.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import (  # noqa: E402
    ALL_GENS, BOUNDARY, FIG, HORIZONS, NAMES, OUT, PRIOR_PREC, SHORT, TAB, TRUTHS,
    eig_tangent, label, row_contrasts, truth,
)  # fmt: skip

OFF = np.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]])
DAUGHTERS = ["†", "1A", "1B", "AA", "AB", "BB"]  # outcome by daughters (same for both rows)
NS = (128, 256, 512, 1024, 2048, 4096)
NWEAK = 4
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
          "#e34948"]  # fmt: skip
TRUTH_SHORT = {"T1_ref": "T1", "T2_toA05": "T2", "T3_Bdiv": "T3", "T4_slowA": "T4",
               "P1": "P1", "P2": "P2", "P3": "P3", "P4": "P4", "T5_C7": "T5"}  # fmt: skip


def load_info(lab: str) -> dict[tuple[str, int], np.ndarray]:
    z = np.load(OUT / f"info_{lab}.npz")
    return {(k.split("_")[0], int(k.split("_")[1])): z[k] for k in z.files if k[1] == "_"}


def design_info(info, design) -> np.ndarray:
    """Sum of I_h over the design's (founder, generation) endpoints (one clone each)."""
    return sum(info[(f, g)] for f, g in design)


BASE = [("A", g) for g in HORIZONS]
DESIGNS = {
    "base 4/6/8/10": BASE,
    "late only 8/10": [("A", 8), ("A", 10)],
    "+ early 1/2/3": [("A", g) for g in (1, 2, 3)] + BASE,
    "early 2/3/4 only": [("A", g) for g in (2, 3, 4)],
    "all gens 1..10": [("A", g) for g in ALL_GENS],
    "B-founded 4/6/8/10": [("B", g) for g in HORIZONS],
    "A + B founded 4/6/8/10": BASE + [("B", g) for g in HORIZONS],
    "A 4/6/8/10 + B 2/4": BASE + [("B", 2), ("B", 4)],
    "A 4/6/8/10 + B 1/2/3": BASE + [("B", g) for g in (1, 2, 3)],
    "A + B founded 1/2/3/4": [(f, g) for f in "AB" for g in (1, 2, 3, 4)],
}


def stable_composition(p12: np.ndarray) -> tuple[float, np.ndarray]:
    """Perron root and stable type distribution u (left Perron vector of the mean matrix)."""
    M = p12.reshape(2, 6) @ OFF
    w, vl = np.linalg.eig(M.T)
    i = int(np.argmax(w.real))
    u = np.abs(vl[:, i].real)
    return float(w.real[i]), u / u.sum()


def swap_basis(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal basis (12, 5) of the parent-swap subspace and its d-basis C (6, 5).

    A swap moves outcome mass between the A and B parents with the population-average
    outcome distribution ``u_A p_A + u_B p_B`` fixed: (A row, B row) = (-u_B d, u_A d)
    with ``d`` a 6-vector over daughter outcomes summing to 0.
    """
    C = np.linalg.qr(np.eye(6) - 1 / 6)[0][:, :5]
    S = np.vstack([-u[1] * C, u[0] * C]) / np.hypot(u[0], u[1])
    return S, C


def mean_span() -> np.ndarray:
    """Orthonormal basis (12, 4) of the tangent directions that change the mean matrix."""
    D = np.zeros((4, 12))
    for k in range(2):
        for j in range(2):
            D[2 * k + j, 6 * k : 6 * k + 6] = OFF[:, j]
    U = row_contrasts()
    return U @ np.linalg.qr((D @ U).T)[0]


def tangent(v: np.ndarray) -> np.ndarray:
    """Project a 12-vector on the tangent space and normalise."""
    U = row_contrasts()
    t = U @ (U.T @ v)
    return t / np.linalg.norm(t)


def n_half(K: np.ndarray, d: np.ndarray, frac: float = 0.5) -> float:
    """Smallest n with posterior variance along unit d <= frac x prior (nuisance free)."""
    U = row_contrasts()
    Kc, dc = U.T @ K @ U, U.T @ d
    lo, hi = 1e-3, 1e8

    def ratio(n):
        return dc @ np.linalg.solve(np.eye(10) + n * Kc, dc)

    if ratio(hi) > frac:
        return np.inf
    for _ in range(100):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if ratio(mid) > frac else (lo, mid)
    return float(hi)


def multinomial_scale(K: np.ndarray, v: np.ndarray, p12: np.ndarray) -> float:
    """Information per clone along v in units of one directly observed fate per row."""
    num = v @ (K * PRIOR_PREC / len(HORIZONS)) @ v
    load = np.abs(v) > 1e-9
    if np.any(load & (p12 <= 0)):
        return 0.0
    return float(num / np.sum(v[load] ** 2 / p12[load]))


# Orthogonal basis of the daughter-outcome contrasts d (6-vectors over †,1A,1B,AA,AB,BB,
# each summing to 0, mutually orthogonal). Used twice: as parent swaps (A row -u_B d,
# B row +u_A d: who does it, population average fixed) and as average-cell changes (A row
# u_A d, B row u_B d). The 10 vectors are an orthonormal basis of the tangent space.
D_BASIS = {
    "AB": ([0, 0, 0, -0.5, 1, -0.5], "mixed AB division", "mixed (AB) vs symmetric divisions"),
    "1A-1B": ([0, 1, -1, 0, 0, 0], "single daughter A vs B",
              "single daughter is A vs B (persist/switch)"),
    "death": ([1, -0.2, -0.2, -0.2, -0.2, -0.2], "death", "death"),
    "AA-BB": ([0, 0, 0, 1, 0, -1], "AA vs BB division", "AA vs BB divisions"),
    "div": ([0, -0.5, -0.5, 1 / 3, 1 / 3, 1 / 3], "division", "divide vs not divide"),
}  # fmt: skip
SWAP_NAMES = {
    "AB": "swap AB: which parent makes the mixed AB divisions",
    "1A-1B": "swap 1A-1B: switching vs persisting, both types",
    "death": "swap death: which type dies",
    "AA-BB": "swap AA-BB: which parent makes AA vs BB",
    "div": "swap div: which type divides",
}
COMMON_NAMES = {
    "AB": "avg AB: mixed vs symmetric divisions",
    "1A-1B": "avg 1A-1B: single daughter A vs B",
    "death": "avg death: death vs survival",
    "AA-BB": "avg AA-BB: AA vs BB divisions",
    "div": "avg div: divide vs not divide",
}


def named_basis(u: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Names and orthonormal basis (12, 10): 5 parent swaps then 5 average-cell changes."""
    names, cols = [], []
    for kind, tab in (("swap", SWAP_NAMES), ("avg", COMMON_NAMES)):
        for key, (d, _, _) in D_BASIS.items():
            d = np.asarray(d, float)
            d = d / np.linalg.norm(d)
            a, b = (-u[1], u[0]) if kind == "swap" else (u[0], u[1])
            v = np.concatenate([a * d, b * d]) / np.hypot(u[0], u[1])
            names.append(tab[key])
            cols.append(v)
    return names, np.array(cols).T


def named_directions(p12: np.ndarray, u: np.ndarray) -> dict[str, np.ndarray]:
    """Named basis directions plus the specific contrasts of Section 6 (unit tangent)."""
    ix = {n: i for i, n in enumerate(NAMES)}

    def vec(**kw):
        v = np.zeros(12)
        for k, x in kw.items():
            v[ix[k.replace("__", ".")]] = x
        return tangent(v)

    nb_names, nb = named_basis(u)
    out = {n: nb[:, i] for i, n in enumerate(nb_names)}
    out.update({
        "B switches back vs B dies": vec(B__to_A=1, B__death=-1),
        "B switches back vs B divides AA": vec(B__to_A=1, B__div_AA=-1),
        "B switches back vs B divides AB": vec(B__to_A=1, B__div_AB=-1),
        "B switches back vs B divides BB": vec(B__to_A=1, B__div_BB=-1),
        "B divides (any) vs not": vec(B__div_AA=1, B__div_AB=1, B__div_BB=1, B__death=-1,
                                      B__to_A=-1, B__persist=-1),
        "B dies vs B persists": vec(B__death=1, B__persist=-1),
        "A dies vs A persists": vec(A__death=1, A__persist=-1),
    })  # fmt: skip
    return out


# ---------------------------------------------------------------------------- analysis
def analyse_truth(lab: str, p12: np.ndarray, B: np.ndarray | None = None) -> dict:
    info = load_info(lab)
    K = design_info(info, BASE) / PRIOR_PREC
    lam, V = eig_tangent(K, B)
    return {"info": info, "K": K, "lam": lam, "V": V, "p": p12}


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    Rm = mean_span()
    res, spec_rows, weak_rows, gen_rows, design_rows, named_rows, btoa_rows = ({}, [], [], [],
                                                                              [], [], [])  # fmt: skip
    for gt in TRUTHS:
        p = truth(gt)
        r = analyse_truth(gt, p)
        rho, u = stable_composition(p)
        S, C = swap_basis(u)
        r.update(rho=rho, u=u, S=S)
        res[gt] = r
        lam, V, K, info = r["lam"], r["V"], r["K"], r["info"]
        spec_rows.append({"truth": gt, "variant": "full", **{f"lam{k + 1}": lam[k]
                                                              for k in range(10)}})  # fmt: skip
        Ibase = design_info(info, BASE)
        nb_names, nb = named_basis(u)
        r["nb"] = nb
        # designs / B founders at the boundary truths: the eps = 0.01 variant (at the exact
        # boundary a B-founded clone larger than 2^(h-1) is impossible unless B divides, so
        # the expected information along B divisions is infinite / floor-dependent)
        dlab = f"{gt}_eps0.01" if gt in BOUNDARY else gt
        dinfo = load_info(dlab)
        dV = eig_tangent(design_info(dinfo, BASE) / PRIOR_PREC)[1]
        for k in range(NWEAK):
            v = V[:, k]
            shares = {f"share_g{g}": float(v @ info[("A", g)] @ v / (v @ Ibase @ v))
                      for g in HORIZONS}  # fmt: skip
            weak_rows.append({
                "truth": gt, "k": k + 1, "lam": lam[k], "n_match_prior": 1 / lam[k],
                "label": label(v, 5), "frac_mean_matrix": float(np.sum((Rm.T @ v) ** 2)),
                "frac_parent_swap": float(np.sum((S.T @ v) ** 2)),
                "frac_B_row": float(np.sum(v[6:] ** 2)),
                "per_clone_vs_one_fate": multinomial_scale(K, v, p),
                **shares, **{nm: v[i] for i, nm in enumerate(NAMES)},
                **{f"nb[{nm.split(':')[0]}]": c for nm, c in zip(nb_names, nb.T @ v,
                                                                strict=True)},
            })  # fmt: skip
            for f in ("A", "B"):
                for g in ALL_GENS:
                    gen_rows.append({"truth": gt, "k": k + 1, "founder": f, "gen": g,
                                     "info_per_clone": float(dV[:, k] @ dinfo[(f, g)]
                                                             @ dV[:, k]) / PRIOR_PREC,
                                     "variant": dlab})  # fmt: skip
        for dn, des in DESIGNS.items():
            Kd = design_info(dinfo, des) / PRIOR_PREC
            ld, Vd = eig_tangent(Kd)
            row = {"truth": gt, "design": dn, "endpoints": len(des)}
            for k in range(NWEAK):
                row[f"lam{k + 1}"] = ld[k]
                row[f"lam{k + 1}_budget"] = ld[k] * len(HORIZONS) / len(des)
            # information the design gives along the BASE weakest directions
            row["lam10_budget"] = ld[-1] * len(HORIZONS) / len(des)
            row["variant"] = dlab
            for k in range(NWEAK):
                v = dV[:, k]
                row[f"base_dir{k + 1}_budget"] = float(v @ Kd @ v) * len(HORIZONS) / len(des)
            design_rows.append(row)
        nd = named_directions(p, u)
        for nm, d in nd.items():
            named_rows.append({
                "truth": gt, "direction": nm, "rayleigh_lam": float(d @ K @ d),
                "n_half": n_half(K, d), "n_quarter": n_half(K, d, 0.25),
                "frac_in_weak4": float(np.sum((V[:, :NWEAK].T @ d) ** 2)),
                "frac_parent_swap": float(np.sum((S.T @ d) ** 2)),
            })  # fmt: skip
        # B.to_A: posterior variance (Gaussian, flat prior) by eigen-direction, ridge direction
        g = tangent(np.eye(12)[NAMES.index("B.to_A")])
        gn = np.eye(12)[NAMES.index("B.to_A")]
        for n in (128, 512, 1024, 4096):
            w = (V.T @ gn) ** 2 / (1 + n * lam)  # variance contributions (x 1/42)
            cov = (V / (1 + n * lam)) @ V.T / PRIOR_PREC
            ridge = cov @ gn
            ridge = ridge / np.linalg.norm(ridge)
            btoa_rows.append({
                "truth": gt, "n": n, "sd_BtoA_gauss": float(np.sqrt(w.sum() / PRIOR_PREC)),
                **{f"share_eig{k + 1}": float(w[k] / w.sum()) for k in range(6)},
                "ridge_label": label(ridge * np.sign(ridge[NAMES.index("B.to_A")]), 5, 0.12),
                "ridge_frac_weak4": float(np.sum((V[:, :NWEAK].T @ ridge) ** 2)),
                "ridge_frac_swap": float(np.sum((S.T @ ridge) ** 2)),
                "n_half_BtoA": n_half(K, g),
                **{f"ridge_{nm}": ridge[i] * np.sign(ridge[NAMES.index("B.to_A")])
                   for i, nm in enumerate(NAMES)},
            })  # fmt: skip

    # ---- boundary variants
    bnd_rows = []
    for gt in BOUNDARY:
        p = truth(gt)
        r0 = res[gt]
        Wfull = r0["V"][:, :NWEAK]
        zero = p == 0
        act = row_contrasts(~zero)
        for var, lab, B in [("full (one-sided)", gt, None), ("face (B never divides)", gt, act),
                            ("eps 0.001", f"{gt}_eps0.001", None),
                            ("eps 0.01", f"{gt}_eps0.01", None)]:  # fmt: skip
            pp = p if lab == gt else np.load(OUT / f"info_{lab}.npz")["p12"]
            r = analyse_truth(lab, pp, B)
            lam, V = r["lam"], r["V"]
            row = {"truth": gt, "variant": var, "dim": len(lam)}
            for k in range(min(len(lam), 10)):
                row[f"lam{k + 1}"] = lam[k]
            for k in range(NWEAK):
                row[f"w{k + 1}_on_zero_outcomes"] = float(np.sum(V[zero, k] ** 2))
                row[f"w{k + 1}_label"] = label(V[:, k], 5)
            if B is None:
                cosines = np.linalg.svd(Wfull.T @ V[:, :NWEAK], compute_uv=False)
                row["overlap_weak4_with_full"] = float(np.mean(cosines**2))
            else:
                # face weak directions inside the full weak-4 subspace?
                cosines = np.linalg.svd(Wfull.T @ V[:, :NWEAK], compute_uv=False)
                row["overlap_weak4_with_full"] = float(np.mean(cosines**2))
                r0f = {"lam": lam, "V": V}
                res[gt]["face"] = r0f
            bnd_rows.append(row)
            if var != "face (B never divides)":
                spec_rows.append({"truth": gt, "variant": var,
                                  **{f"lam{k + 1}": lam[k] for k in range(10)}})  # fmt: skip
            else:
                spec_rows.append({"truth": gt, "variant": var,
                                  **{f"lam{k + 1}": lam[k] for k in range(len(lam))}})  # fmt: skip
        # face: multinomial scale for the face's weakest directions
        fr = res[gt]["face"]
        for k in range(NWEAK):
            bnd_rows[-3]["w%d_face_per_clone_vs_one_fate" % (k + 1)] = multinomial_scale(
                res[gt]["K"], fr["V"][:, k], p)

    # ---- recurrence: subspace overlap of the weak-4 subspaces across truths
    rec = np.zeros((len(TRUTHS), len(TRUTHS)))
    for i, a in enumerate(TRUTHS):
        for j, b in enumerate(TRUTHS):
            s = np.linalg.svd(res[a]["V"][:, :NWEAK].T @ res[b]["V"][:, :NWEAK],
                              compute_uv=False)  # fmt: skip
            rec[i, j] = np.mean(s**2)
    # individual-vector |cos| (weakest 4 x weakest 4)
    vecs = np.hstack([res[gt]["V"][:, :NWEAK] for gt in TRUTHS])
    cosm = np.abs(vecs.T @ vecs)
    # random-subspace baseline for overlap of two 4-dim subspaces in 10 dims: 4/10
    pd.DataFrame(rec, index=TRUTHS, columns=TRUTHS).to_csv(TAB / "recurrence_overlap.csv")
    pd.DataFrame(cosm, index=[f"{TRUTH_SHORT[g]}w{k + 1}" for g in TRUTHS for k in range(NWEAK)],
                 columns=[f"{TRUTH_SHORT[g]}w{k + 1}" for g in TRUTHS
                          for k in range(NWEAK)]).to_csv(TAB / "recurrence_cosines.csv")  # fmt: skip

    # ---- write tables
    def write(df: pd.DataFrame, name: str, fmt: str = ".4g") -> None:
        df.to_csv(TAB / f"{name}.csv", index=False)
        (TAB / f"{name}.md").write_text(df.to_markdown(index=False, floatfmt=fmt) + "\n")

    spec = pd.DataFrame(spec_rows)
    write(spec, "spectra")
    weak = pd.DataFrame(weak_rows)
    write(weak, "weak_directions", ".3g")
    write(pd.DataFrame(gen_rows), "generation_info")
    write(pd.DataFrame(design_rows), "designs", ".3g")
    write(pd.DataFrame(named_rows), "named_directions", ".3g")
    write(pd.DataFrame(btoa_rows), "btoa_variance", ".3g")
    write(pd.DataFrame(bnd_rows), "boundary", ".3g")
    np.savez(OUT / "fisher_eigen.npz",
             **{f"{gt}_lam": res[gt]["lam"] for gt in TRUTHS},
             **{f"{gt}_V": res[gt]["V"] for gt in TRUTHS},
             **{f"{gt}_K": res[gt]["K"] for gt in TRUTHS},
             **{f"{gt}_u": res[gt]["u"] for gt in TRUTHS})  # fmt: skip

    # swap-subspace eigen (d-vectors) per truth: the "who does it" directions
    swap_rows = []
    for gt in TRUTHS:
        S, C = swap_basis(res[gt]["u"])
        ls, c = np.linalg.eigh(S.T @ res[gt]["K"] @ S)
        for k in range(5):
            d = C @ c[:, k]
            d *= np.sign(d[np.argmax(np.abs(d))])
            swap_rows.append({"truth": gt, "k": k + 1, "lam": ls[k], "n_match_prior": 1 / ls[k],
                              "u_A": res[gt]["u"][0],
                              **{f"d_{n}": x for n, x in zip(DAUGHTERS, d, strict=True)}})  # fmt: skip
        # orthogonal complement ("average cell")
        U = row_contrasts()
        Sc = U @ np.linalg.svd(S.T @ U)[2][5:].T
        lc, _ = eig_tangent(res[gt]["K"], Sc)
        for k in range(5):
            swap_rows.append({"truth": gt, "k": 6 + k, "lam": lc[k], "n_match_prior": 1 / lc[k],
                              "u_A": res[gt]["u"][0]})  # fmt: skip
    write(pd.DataFrame(swap_rows), "swap_subspace", ".3g")

    plots(res, weak, pd.DataFrame(design_rows), pd.DataFrame(gen_rows), rec, spec)
    print("done")


# ---------------------------------------------------------------------------- figures
def _style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 9.5,
        "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
        "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0",
        "grid.linewidth": 0.6, "legend.frameon": False, "figure.facecolor": "white",
    })  # fmt: skip


def diverging():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "bgr", ["#104281", "#3987e5", "#cde2fb", "#f0efec", "#f6c9c4", "#e34948", "#8f1d1c"])


def plots(res, weak, designs, gens, rec, spec) -> None:
    import matplotlib.pyplot as plt

    _style()
    sh = [TRUTH_SHORT[g] for g in TRUTHS]
    # (a) spectra
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for n in NS:
        ax.axhline(1 / n, color="#c3c2b7", lw=0.8, ls="--", zorder=0)
        ax.text(len(TRUTHS) - 0.35, 1 / n, f"n={n}", va="center", ha="left", fontsize=7,
                color="#52514e")  # fmt: skip
    for i, gt in enumerate(TRUTHS):
        lam = res[gt]["lam"]
        cols = ["#e34948" if k < NWEAK else "#2a78d6" for k in range(10)]
        ax.scatter(np.full(10, i), lam, s=22, c=cols, edgecolor="white", lw=0.6, zorder=3)
        if gt in BOUNDARY:
            lf = res[gt]["face"]["lam"]
            ax.scatter(np.full(len(lf), i + 0.28), lf, s=14, marker="D", facecolor="none",
                       edgecolor="#6250d6", lw=0.8, zorder=3)  # fmt: skip
    ax.set_yscale("log")
    ax.set_xticks(range(len(TRUTHS)), sh)
    ax.set_xlim(-0.6, len(TRUTHS) + 0.4)
    ax.set_ylabel("prior-whitened eigenvalue λ\n(information of 1 clone per endpoint / prior)")
    ax.set_title("Sloppiness spectra (design 4/6/8/10, A founder). Below the dashed line λ = 1/n"
                 "\nthe data at n clones per endpoint carry less information than the flat prior",
                 fontsize=8.5, loc="left")  # fmt: skip
    from matplotlib.lines import Line2D

    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color="#e34948", label="4 weakest"),
        Line2D([], [], marker="o", ls="", color="#2a78d6", label="others"),
        Line2D([], [], marker="D", ls="", mfc="none", color="#6250d6",
               label="face (B never divides), T1/T2/T4"),
    ], loc="upper left", fontsize=7.5, ncol=3)  # fmt: skip
    ax.set_ylim(1.5e-4, 80)
    fig.tight_layout()
    fig.savefig(FIG / "spectra.png")
    plt.close(fig)

    # (b) loadings heatmap: 12 probabilities x (truth, k)
    cols = [(gt, k) for gt in TRUTHS for k in range(NWEAK)]
    L = np.array([res[gt]["V"][:, k] for gt, k in cols]).T
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.grid(False)
    im = ax.imshow(L, cmap=diverging(), vmin=-0.85, vmax=0.85, aspect="auto")
    ax.set_yticks(range(12), [f"{s}  ({n})" for s, n in zip(SHORT, NAMES, strict=True)],
                  fontsize=7.2)  # fmt: skip
    ax.set_xticks(range(len(cols)), [f"{1 / res[gt]['lam'][k]:.0f}" for gt, k in cols],
                  fontsize=6, rotation=90)  # fmt: skip
    for i in range(1, len(TRUTHS)):
        ax.axvline(i * NWEAK - 0.5, color="white", lw=2.2)
    ax.axhline(5.5, color="white", lw=2.2)
    for i, gt in enumerate(TRUTHS):
        ax.text(i * NWEAK + 1.5, -1.0, TRUTH_SHORT[gt], ha="center", va="bottom", fontsize=8,
                fontweight="bold")  # fmt: skip
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("loading (unit vector)", fontsize=7.5)
    ax.set_xlabel("4 weakest eigen-directions per truth (weakest first), labelled by 1/λ ="
                  " clones per endpoint at which data = prior", fontsize=7.5)  # fmt: skip
    fig.tight_layout()
    fig.savefig(FIG / "loadings.png")
    plt.close(fig)

    # (b2) loadings as 2 x 6 (parent row x daughter outcome) mini-panels: parent swaps show
    # as columns with opposite signs
    fig, axes = plt.subplots(len(TRUTHS), NWEAK, figsize=(7.2, 9.6))
    for i, gt in enumerate(TRUTHS):
        for k in range(NWEAK):
            a = axes[i, k]
            a.grid(False)
            a.imshow(res[gt]["V"][:, k].reshape(2, 6), cmap=diverging(), vmin=-0.85, vmax=0.85)
            a.set_xticks(range(6), DAUGHTERS if i == len(TRUTHS) - 1 else [], fontsize=6.5)
            a.set_yticks([0, 1], ["A", "B"] if k == 0 else [], fontsize=7)
            for (yy, xx), val in np.ndenumerate(res[gt]["V"][:, k].reshape(2, 6)):
                if abs(val) >= 0.2:
                    a.text(xx, yy, f"{val:+.1f}".replace("0.", "."), ha="center",
                           va="center", fontsize=5.5,
                           color="white" if abs(val) > 0.5 else "#0b0b0b")  # fmt: skip
            sw = weak[(weak.truth == gt) & (weak.k == k + 1)].frac_parent_swap.iloc[0]
            a.set_title(f"{TRUTH_SHORT[gt]} w{k + 1}: 1/λ={1 / res[gt]['lam'][k]:.0f}, "
                        f"swap {sw:.0%}", fontsize=6.5, pad=2)  # fmt: skip
            for s in a.spines.values():
                s.set_visible(False)
    fig.suptitle("Weakest directions as parent (row) × daughters (column); opposite signs in "
                 "one column = a parent swap", fontsize=8.5)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(FIG / "loadings_grid.png")
    plt.close(fig)

    # (b3) composition of the weakest directions in the named basis
    nbn = [n for n in SWAP_NAMES.values()] + [n for n in COMMON_NAMES.values()]
    Cm = np.array([res[gt]["nb"].T @ res[gt]["V"][:, k] for gt, k in cols]).T
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ax.grid(False)
    im = ax.imshow(Cm, cmap=diverging(), vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(10), nbn, fontsize=6.8)
    ax.set_xticks(range(len(cols)), [f"{1 / res[gt]['lam'][k]:.0f}" for gt, k in cols],
                  fontsize=6, rotation=90)  # fmt: skip
    for i in range(1, len(TRUTHS)):
        ax.axvline(i * NWEAK - 0.5, color="white", lw=2.2)
    ax.axhline(4.5, color="white", lw=2.2)
    for i, gt in enumerate(TRUTHS):
        ax.text(i * NWEAK + 1.5, -0.8, TRUTH_SHORT[gt], ha="center", va="bottom", fontsize=8,
                fontweight="bold")  # fmt: skip
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("coefficient (squares sum to 1)", fontsize=7)
    ax.set_xlabel("4 weakest eigen-directions per truth, labelled by 1/λ", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "named_composition.png")
    plt.close(fig)

    # (c) generation contributions + designs
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1.1, 1.1,
                                                                                    1.5]})  # fmt: skip
    ax = axes[0]
    w = weak[weak.k == 1].set_index("truth").loc[TRUTHS]
    bottom = np.zeros(len(TRUTHS))
    for j, g in enumerate(HORIZONS):
        vals = w[f"share_g{g}"].to_numpy()
        ax.bar(range(len(TRUTHS)), vals, bottom=bottom, color=SERIES[j], width=0.78,
               edgecolor="white", lw=1, label=f"gen {g}")  # fmt: skip
        bottom += vals
    ax.set_xticks(range(len(TRUTHS)), sh, fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_title("share of the information along\nthe weakest direction, by endpoint",
                 fontsize=8)  # fmt: skip
    ax.legend(fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    gg = gens[gens.k == 1]
    for i, gt in enumerate(TRUTHS):
        for f, ls in (("A", "-"), ("B", ":")):
            s = gg[(gg.truth == gt) & (gg.founder == f)]
            ax.plot(s.gen, s.info_per_clone, ls=ls, color=SERIES[i % 8] if i < 8 else "#52514e",
                    lw=1.3)  # fmt: skip
    ax.set_yscale("log")
    ax.set_xlabel("generation of the endpoint")
    ax.set_title("information per clone along the\nweakest direction (— A, ··· B founder)*",
                 fontsize=8)  # fmt: skip
    ax.set_xticks(range(1, 11, 3))
    ax = axes[2]
    dnames = list(DESIGNS)
    for i, gt in enumerate(TRUTHS):
        s = designs[designs.truth == gt].set_index("design").loc[dnames]
        ax.plot(s.lam1_budget.to_numpy() / s.lam1_budget.iloc[0], range(len(dnames)), "o",
                color=SERIES[i % 8] if i < 8 else "#52514e", ms=4.5, alpha=0.9,
                label=TRUTH_SHORT[gt])  # fmt: skip
    ax.axvline(1, color="#c3c2b7", lw=0.8)
    ax.set_xscale("log")
    ax.set_yticks(range(len(dnames)), dnames, fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlabel("weakest λ vs base design\n(same total number of clones)", fontsize=7.5)
    ax.set_title("counterfactual designs", fontsize=8)
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=9, fontsize=7, bbox_to_anchor=(0.6, 0.025),
               handletextpad=0.2, columnspacing=0.9)  # fmt: skip
    fig.text(0.01, 0.003, "* T1/T2/T4 B-founded curves and all designs at the eps = 0.01 "
             "variant (see boundary)", fontsize=6, color="#52514e")  # fmt: skip
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "generations_designs.png")
    plt.close(fig)

    # recurrence heatmap
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.grid(False)
    im = ax.imshow(rec, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(TRUTHS)), sh, fontsize=7)
    ax.set_yticks(range(len(TRUTHS)), sh, fontsize=7)
    for (i, j), val in np.ndenumerate(rec):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.5,
                color="white" if val > 0.6 else "#0b0b0b")  # fmt: skip
    fig.colorbar(im, ax=ax, fraction=0.045)
    ax.set_title("overlap of the weak-4 subspaces\n(mean cos² of principal angles; random 0.4)",
                 fontsize=7.5)  # fmt: skip
    fig.tight_layout()
    fig.savefig(FIG / "recurrence.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
