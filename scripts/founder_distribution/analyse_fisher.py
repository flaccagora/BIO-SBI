"""Information study of the founder (seeding) distribution: known vs estimated.

Reads ``fisher.py``'s cache (15 x 15 information per detected clone, per truth, generation
and founder distribution). For each truth, design (XG2 endpoints 4/6/8/10; R1's time
points 1/2/10 with shares 1/4, 5/8, 1/8, founders unlabelled) and founder distribution:

* K  - founder distribution known;
* E  - p_A and the founder-size probabilities on the true support estimated (nuisances,
       Schur complement); E_type (only p_A estimated) and E_size (only sizes) decompose it.

Reported: weakest prior-whitened eigenvalue lam_min (1/lam_min clones), asymptotic sd of
B->A, m_BA, growth rate rho, stable A fraction at N = 2048 clones (512 per XG2 endpoint),
and, for E, the sd of p_A and of the mean founder size (jointly with the transitions, and
with the transitions known). Multiple conditions: shared transitions, a separate founder
distribution (estimated) per condition, equal split of the same N clones.

  uv run python scripts/founder_distribution/analyse_fisher.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fd_common as F  # noqa: E402
from fd_common import od  # noqa: E402

MODES = ("K", "E", "E_type", "E_size")


def face_contrasts(pi: np.ndarray) -> np.ndarray:
    """Orthonormal contrasts (5, k) of the sizes with pi_m > 0 (k = support - 1)."""
    idx = np.flatnonzero(pi > 0)
    if len(idx) < 2:
        return np.zeros((5, 0))
    M = np.zeros((5, len(idx)))
    M[idx, np.arange(len(idx))] = 1.0
    M[idx] -= 1.0 / len(idx)
    q, _ = np.linalg.qr(M)
    return q[:, : len(idx) - 1]


def to_face(M: np.ndarray, pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Information in (10 transitions, p_A, size contrasts on the true support)."""
    W = face_contrasts(pi)
    T = F.V_PI.T @ W  # V coordinates of the face contrasts
    k = W.shape[1]
    J = np.zeros((15, 11 + k))
    J[:11, :11] = np.eye(11)
    J[11:, 11:] = T
    return J.T @ M @ J, W


def mode_nuisance(mode: str, k: int) -> list[int]:
    return {"K": [], "E": [10, *range(11, 11 + k)], "E_type": [10],
            "E_size": list(range(11, 11 + k))}[mode]  # fmt: skip


def design_info(z: dict, dist: str, design: dict) -> np.ndarray:
    i = list(z["dists"]).index(dist)
    gens = list(z["gens"])
    return sum(s * z["info"][i, gens.index(h)] for h, s in design.items())


def summarise(I_th: np.ndarray, cs: dict, n: float = F.N_TOTAL) -> dict:
    lam = F.lam_min(I_th / od.PRIOR_PREC)
    out = {"lam_min": lam, "inv_lam_min": 1 / lam if lam > 0 else np.inf}
    for f, c in cs.items():
        out[f"sd_{f}"] = F.sd_functional(I_th, c, n)
        # with the flat Dirichlet prior (precision 42 per tangent direction), Gaussian approx.
        out[f"sdprior_{f}"] = float(np.sqrt(c @ np.linalg.solve(
            n * I_th + od.PRIOR_PREC * np.eye(len(c)), c)))  # fmt: skip
    return out


def founder_sds(Mf: np.ndarray, W: np.ndarray, n: float = F.N_TOTAL) -> dict:
    """Sd of p_A and of the mean founder size: jointly (transitions estimated) and with the
    transitions known.
    """
    k = W.shape[1]
    g = W.T @ F.SIZES.astype(float)  # d mean size / d face coordinates
    out = {}
    for tag, idx in (("joint", list(range(11 + k))), ("th_known", list(range(10, 11 + k)))):
        A = Mf[np.ix_(idx, idx)]
        try:
            inv = np.linalg.inv(A) / n
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(A) / n
        o = idx.index(10)
        out[f"sd_pA_{tag}"] = float(np.sqrt(max(inv[o, o], 0)))
        if k:
            B = inv[o + 1 :, o + 1 :]
            out[f"sd_mean_size_{tag}"] = float(np.sqrt(max(g @ B @ g, 0)))
        else:
            out[f"sd_mean_size_{tag}"] = np.nan
    return out


def main_table() -> pd.DataFrame:
    rows = []
    for gt in F.TRUTHS:
        z = dict(np.load(F.CACHE / f"{gt}.npz"))
        cs = F.functional_gradients(z["p12"], z["U"])
        for dname, design in F.DESIGNS.items():
            for dist in z["dists"]:
                pi, pa = F.all_dists()[str(dist)]
                Mf, W = to_face(design_info(z, str(dist), design), pi)
                k = W.shape[1]
                base = {"truth": F.SHORTNAME[gt], "design": dname, "dist": str(dist),
                        "size": str(dist).split("_")[0], "pA": pa}  # fmt: skip
                fs = founder_sds(Mf, W)
                for mode in MODES:
                    nu = mode_nuisance(mode, k)
                    if mode == "E_size" and k == 0:
                        continue
                    I_th = F.schur(Mf, np.arange(10), np.array(nu, int))
                    r = {**base, "mode": mode, **summarise(I_th, cs)}
                    if mode == "E":
                        r.update(fs)
                    rows.append(r)
    df = pd.DataFrame(rows)
    # gains relative to the single-A-founder known reference (same truth): XG2 design, and
    # the same design
    for ref_design in ("XG2", "same"):
        sel = (df.dist == "S1_pA1.0") & (df["mode"] == "K")
        ref = df[sel].set_index(["truth", "design"])
        key = list(zip(df.truth, df.design if ref_design == "same" else ["XG2"] * len(df),
                       strict=True))  # fmt: skip
        tag = "xg2" if ref_design == "XG2" else "design"
        df[f"gain_lam_vs_{tag}"] = df.lam_min.values / ref.loc[key, "lam_min"].values
        for f in F.FUNCS:
            df[f"gain_var_{f}_vs_{tag}"] = (ref.loc[key, f"sd_{f}"].values
                                            / df[f"sd_{f}"].values) ** 2  # fmt: skip
    return df


# ---------------------------------------------------------------------- multiple conditions
COMBOS = {
    "S2: 0.9 + 0.3": [("S2_pA0.9", 0.5), ("S2_pA0.3", 0.5)],
    "S2: 0.9 + 0.6 + 0.3": [("S2_pA0.9", 1 / 3), ("S2_pA0.6", 1 / 3), ("S2_pA0.3", 1 / 3)],
    "S2: 1.0 + 0.0 (sorted A / sorted B, purity unknown)": [("S2_pA1.0", 0.5), ("S2_pA0.0", 0.5)],
    "S1: 0.9 + 0.3": [("S1_pA0.9", 0.5), ("S1_pA0.3", 0.5)],
    "S4: 0.9 + 0.3": [("S4_pA0.9", 0.5), ("S4_pA0.3", 0.5)],
    "p_A 0.7: S1 + S4": [("S1_pA0.7", 0.5), ("S4_pA0.7", 0.5)],
}
SINGLES = {  # single-condition comparisons with the same N
    "S2: 0.9 + 0.3": ["S2_pA0.9", "S2_pA0.6", "S2_pA0.3"],
    "S2: 0.9 + 0.6 + 0.3": ["S2_pA0.9", "S2_pA0.6", "S2_pA0.3"],
    "S2: 1.0 + 0.0 (sorted A / sorted B, purity unknown)": ["S2_pA1.0", "S2_pA0.5", "S2_pA0.0"],
    "S1: 0.9 + 0.3": ["S1_pA0.9", "S1_pA0.6", "S1_pA0.3"],
    "S4: 0.9 + 0.3": ["S4_pA0.9", "S4_pA0.6", "S4_pA0.3"],
    "p_A 0.7: S1 + S4": ["S1_pA0.7", "S4_pA0.7"],
}


def joint_conditions(z: dict, parts: list, design: dict) -> tuple[np.ndarray, list]:
    """Joint information (shared transitions + per-condition nuisances) per clone."""
    blocks = []
    for dist, w in parts:
        pi, _ = F.all_dists()[dist]
        Mf, W = to_face(design_info(z, dist, design), pi)
        blocks.append((w * Mf, W))
    nn = [b.shape[0] - 10 for b, _ in blocks]
    D = 10 + sum(nn)
    J = np.zeros((D, D))
    off = 10
    for (Mb, _), k in zip(blocks, nn, strict=True):
        idx = list(range(10)) + list(range(off, off + k))
        J[np.ix_(idx, idx)] += Mb
        off += k
    return J, [(W, k) for (_, W), k in zip(blocks, nn, strict=True)]


def multi_table(df_main: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gt in F.TRUTHS:
        z = dict(np.load(F.CACHE / f"{gt}.npz"))
        cs = F.functional_gradients(z["p12"], z["U"])
        for dname, design in F.DESIGNS.items():
            for name, parts in COMBOS.items():
                J, meta = joint_conditions(z, parts, design)
                nuis = np.arange(10, J.shape[0])
                for mode, I_th in (("E", F.schur(J, np.arange(10), nuis)),
                                   ("K", J[:10, :10])):  # fmt: skip
                    r = {"truth": F.SHORTNAME[gt], "design": dname, "combo": name, "mode": mode,
                         **summarise(I_th, cs)}  # fmt: skip
                    if mode == "E":
                        inv = np.linalg.pinv(J) / F.N_TOTAL
                        off = 10
                        sds = []
                        for (W, k), (dist, _) in zip(meta, parts, strict=True):
                            sds.append(f"{np.sqrt(max(inv[off, off], 0)):.3f}")
                            off += k
                        r["sd_pA_per_condition"] = " / ".join(sds)
                    rows.append(r)
                for dist in SINGLES[name]:
                    s = df_main[(df_main.truth == F.SHORTNAME[gt]) & (df_main.design == dname)
                                & (df_main.dist == dist) & (df_main["mode"] == "E")].iloc[0]  # fmt: skip
                    rows.append({"truth": F.SHORTNAME[gt], "design": dname, "combo": name,
                                 "mode": f"single {dist} E", "lam_min": s.lam_min,
                                 "inv_lam_min": s.inv_lam_min,
                                 **{f"sd_{f}": s[f"sd_{f}"] for f in F.FUNCS},
                                 "sd_pA_per_condition": f"{s.sd_pA_joint:.3f}"})  # fmt: skip
    return pd.DataFrame(rows)


def multi_ratios(mt: pd.DataFrame) -> pd.DataFrame:
    """Median over truths of combined / single-condition ratios (lam_min; 1/Var(B->A))."""
    rows = []
    for (design, combo), s in mt.groupby(["design", "combo"], sort=False):
        E = s[s["mode"] == "E"].set_index("truth")
        K = s[s["mode"] == "K"].set_index("truth")
        r = {"design": design, "conditions (estimated)": combo,
             "E/K λ": float(np.median(E.lam_min / K.lam_min))}  # fmt: skip
        for i, mo in enumerate(m for m in s["mode"].unique() if m.startswith("single")):
            S = s[s["mode"] == mo].set_index("truth")
            r[f"vs single {i + 1}"] = mo.split()[1].replace("_pA", " p_A ")
            r[f"λ ratio {i + 1}"] = float(np.median(E.lam_min / S.lam_min))
            r[f"Var(B→A) ratio {i + 1}"] = float(np.median((S["sd_B.to_A"] / E["sd_B.to_A"]) ** 2))
        grp = np.where(E.index.str.startswith("P"), "P", "T")
        for g in ("T", "P"):
            v = [np.array(x.split(" / "), float) for x in E.sd_pA_per_condition[grp == g]]
            r[f"sd p_A per condition ({g})"] = " / ".join(f"{x:.2f}" for x in np.median(v, 0))
        rows.append(r)
    return pd.DataFrame(rows)


def md(df: pd.DataFrame, path: Path, floatfmt=".3g") -> None:
    path.write_text(df.to_markdown(index=False, floatfmt=floatfmt) + "\n")


KEY_DISTS = ["S1_pA1.0", "S1_pA0.9", "S1_pA0.7", "S1_pA0.5", "S1_pA0.3", "S1_pA0.0", "S2_pA1.0",
             "S2_pA0.9", "S2_pA0.7", "S2_pA0.5", "S3_pA0.9", "S3_pA0.5", "S4_pA0.9", "S4_pA0.5"]


def key_table(df: pd.DataFrame) -> pd.DataFrame:
    """Report table: median over truths (and T / P groups) of the main gains and sds."""
    d = df[df.dist.isin(KEY_DISTS)].copy()
    d["grp"] = np.where(d.truth.str.startswith("P"), "P", "T")
    rows = []
    for (design, dist), s in d.groupby(["design", "dist"], sort=False):
        K, E = s[s["mode"] == "K"], s[s["mode"] == "E"]
        r = {"design": design, "founders": dist.replace("_pA", ", p_A ")}
        for m, x in (("K", K), ("E", E)):
            r[f"λ gain {m}"] = x.gain_lam_vs_design.median()
            r[f"Var(B→A) gain {m}"] = x["gain_var_B.to_A_vs_design"].median()
        r["Var(m_BA) gain E"] = E["gain_var_m_BA_vs_design"].median()
        r["Var(ρ) gain E"] = E["gain_var_rho_vs_design"].median()
        r["E/K λ (range over truths)"] = (f"{(E.lam_min.values / K.lam_min.values).min():.2f}–"
                                          f"{(E.lam_min.values / K.lam_min.values).max():.2f}")  # fmt: skip
        r["sd p_A: T / P truths"] = (f"{E[E.grp == 'T'].sd_pA_joint.median():.2f} / "
                                     f"{E[E.grp == 'P'].sd_pA_joint.median():.2f}")  # fmt: skip
        r["sd mean size: T / P"] = (f"{E[E.grp == 'T'].sd_mean_size_joint.median():.2f} / "
                                    f"{E[E.grp == 'P'].sd_mean_size_joint.median():.2f}")  # fmt: skip
        rows.append(r)
    out = pd.DataFrame(rows)
    order = {d: i for i, d in enumerate(KEY_DISTS)}
    out["o"] = [order[f.replace(", p_A ", "_pA")] for f in out.founders]
    return out.sort_values(["design", "o"], ascending=[False, True]).drop(columns="o")


def main() -> None:
    F.TAB.mkdir(parents=True, exist_ok=True)
    df = main_table()
    df.to_csv(F.TAB / "fisher_main.csv", index=False)
    kt = key_table(df)
    kt.to_csv(F.TAB / "fisher_key.csv", index=False)
    md(kt, F.TAB / "fisher_key.md", floatfmt=".2f")
    # summary over truths: median gains (vs the single-A known reference on XG2)
    g = df.groupby(["design", "size", "pA", "mode"], sort=False)
    summ = g.agg(
        gain_lam_med=("gain_lam_vs_xg2", "median"), gain_lam_min=("gain_lam_vs_xg2", "min"),
        gain_lam_max=("gain_lam_vs_xg2", "max"),
        gain_var_BtoA_med=("gain_var_B.to_A_vs_xg2", "median"),
        gain_var_BtoA_min=("gain_var_B.to_A_vs_xg2", "min"),
        gain_var_BtoA_max=("gain_var_B.to_A_vs_xg2", "max"),
        gain_var_mBA_med=("gain_var_m_BA_vs_xg2", "median"),
        gain_var_rho_med=("gain_var_rho_vs_xg2", "median"),
        gain_var_stableA_med=("gain_var_stableA_vs_xg2", "median"),
        sd_pA_med=("sd_pA_joint", "median"), sd_pA_max=("sd_pA_joint", "max"),
        sd_mean_size_med=("sd_mean_size_joint", "median"),
        sd_pA_thknown_med=("sd_pA_th_known", "median"),
    ).reset_index()  # fmt: skip
    summ.to_csv(F.TAB / "fisher_summary.csv", index=False)
    md(summ[summ["mode"].isin(["K", "E"])], F.TAB / "fisher_summary.md")
    # per truth, the main comparisons (XG2 design): K vs E for four representative dists
    rep = ["S1_pA1.0", "S1_pA0.7", "S2_pA0.7", "S3_pA0.9", "S4_pA0.5", "S2_pA0.3"]
    pt = df[(df.dist.isin(rep)) & (df["mode"].isin(["K", "E"]))]
    cols = ["truth", "design", "dist", "mode", "inv_lam_min", "sd_B.to_A", "sd_m_BA", "sd_rho",
            "sd_stableA", "sd_pA_joint", "sd_mean_size_joint", "sd_pA_th_known"]  # fmt: skip
    pt[cols].to_csv(F.TAB / "fisher_per_truth.csv", index=False)
    md(pt[pt.design == "XG2"][cols], F.TAB / "fisher_per_truth_xg2.md")
    md(pt[pt.design == "R1t"][cols], F.TAB / "fisher_per_truth_r1t.md")
    mt = multi_table(df)
    mt.to_csv(F.TAB / "multi_conditions.csv", index=False)
    mg = mt.groupby(["design", "combo", "mode"], sort=False).agg(
        inv_lam_med=("inv_lam_min", "median"), sd_BtoA_med=("sd_B.to_A", "median"),
        sd_mBA_med=("sd_m_BA", "median"), sd_rho_med=("sd_rho", "median")).reset_index()  # fmt: skip
    mg.to_csv(F.TAB / "multi_conditions_summary.csv", index=False)
    md(mg, F.TAB / "multi_conditions_summary.md")
    mr = multi_ratios(mt)
    mr.to_csv(F.TAB / "multi_conditions_ratios.csv", index=False)
    md(mr, F.TAB / "multi_conditions_ratios.md", floatfmt=".2f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(summ[summ["mode"].isin(["K", "E"])].round(3).to_string())
    print(mg.round(4).to_string())


if __name__ == "__main__":
    main()
