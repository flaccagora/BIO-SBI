"""Lab knowledge as an informative founder prior: information between "estimated" and "known".

For every truth, design and founder distribution of the Fisher study (``fisher.py`` cache),
the founder parameters (p_A and the founder-size probabilities on the true support) get the
prior of the ``lab<k>`` models of ``run.py``: Beta(1 + k p_A, 1 + k (1 - p_A)) and
Dirichlet(1 + k pi), mode at the truth, in a Gaussian approximation (Beta / Dirichlet
covariance, size covariance projected on the orthonormal face contrasts). The prior precision
is added to the nuisance block of the information of N = 2048 clones before the nuisances
are profiled out (Schur complement), so k = 0 ... infinity moves from E (no prior on the
founder parameters) to K (known founder distribution).

Reported per k: gain on var(B->A) and on the total variance of the 12 probabilities against
single known A founders on the same design, and the share of the E -> K loss recovered,
``(gain_k - gain_E) / (gain_K - gain_E)``.

  uv run python scripts/founder_distribution/lab_prior_fisher.py

Writes ``docs/founder_distribution/tables/lab_prior_fisher{,_summary}.csv`` and the figure
``docs/founder_distribution/figures/lab_prior.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_fisher as AF  # noqa: E402
import fd_common as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "docs" / "founder_distribution" / "tables"
FIG = ROOT / "docs" / "founder_distribution" / "figures"
KAPPAS = (3, 10, 20, 50, 100, 300, 1000)
SIZES = ("S1", "S2", "S3", "S4")


def prior_precision(pi: np.ndarray, pa: float, W: np.ndarray, k: float) -> np.ndarray:
    """Precision (1 + kdim) of the lab prior in (p_A, face contrasts), Gaussian approx."""
    a, b = 1 + k * pa, 1 + k * (1 - pa)
    var_pa = a * b / ((a + b) ** 2 * (a + b + 1))
    P = np.zeros((1 + W.shape[1],) * 2)
    P[0, 0] = 1 / var_pa
    if W.shape[1]:
        alpha = np.where(pi > 0, 1 + k * pi, 0.0)
        mu = alpha / alpha.sum()
        S = (np.diag(mu) - np.outer(mu, mu)) / (alpha.sum() + 1)
        P[1:, 1:] = np.linalg.inv(W.T @ S @ W)
    return P


def info_with_prior(Mf: np.ndarray, P: np.ndarray | None, n: float = F.N_TOTAL) -> np.ndarray:
    """Per-clone transition information with the founder nuisances profiled out, after
    adding the prior precision ``P`` (whole sample) to their block; None = known."""
    if P is None:
        return Mf[:10, :10]
    M = Mf.copy()
    M[10:, 10:] += P / n
    return F.schur(M, np.arange(10), np.arange(10, M.shape[0]))


def table() -> pd.DataFrame:
    rows = []
    for gt in F.TRUTHS:
        z = dict(np.load(F.CACHE / f"{gt}.npz"))
        c = F.functional_gradients(z["p12"], z["U"])["B.to_A"]
        for dname, design in F.DESIGNS.items():
            for dist in z["dists"]:
                pi, pa = F.all_dists()[str(dist)]
                Mf, W = AF.to_face(AF.design_info(z, str(dist), design), pi)
                cases = [("E", 0.0, np.zeros((Mf.shape[0] - 10,) * 2)), ("K", np.inf, None)]
                cases += [(f"lab{k}", float(k), prior_precision(pi, pa, W, k)) for k in KAPPAS]
                for name, k, P in cases:
                    I_th = info_with_prior(Mf, P)
                    inv = np.linalg.pinv(I_th, hermitian=True)
                    rows.append({"truth": F.SHORTNAME[gt], "design": dname, "dist": str(dist),
                                 "size": str(dist).split("_")[0], "pA": pa, "prior": name,
                                 "kappa": k, "var_BtoA": float(c @ inv @ c) / F.N_TOTAL,
                                 "trvar12": float(np.trace(inv)) / F.N_TOTAL})  # fmt: skip
    df = pd.DataFrame(rows)
    ref = df[(df.dist == "S1_pA1.0") & (df.prior == "K")].set_index(["truth", "design"])
    key = list(zip(df.truth, df.design, strict=True))
    df["gain_BtoA"] = ref.loc[key, "var_BtoA"].values / df.var_BtoA.values
    df["gain_all12"] = ref.loc[key, "trvar12"].values / df.trvar12.values
    idx = ["truth", "design", "dist"]
    k3 = list(zip(df.truth, df.design, df.dist, strict=True))
    for g in ("gain_BtoA", "gain_all12"):
        e = df[df.prior == "E"].set_index(idx)[g].loc[k3].values
        kk = df[df.prior == "K"].set_index(idx)[g].loc[k3].values
        span = kk - e
        with np.errstate(invalid="ignore", divide="ignore"):
            df[g.replace("gain", "recovered")] = np.where(
                np.abs(span) > 0.02 * np.abs(kk), (df[g].values - e) / span, np.nan)
    return df


def figure(df: pd.DataFrame) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    cols = {"S1": "#1c5cab", "S2": "#3a9e6e", "S3": "#e0a030", "S4": "#b8302f"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), constrained_layout=True, sharex=True)
    xs = [0.6, *KAPPAS, 3000]
    def sd_pa(k: float, p: float = 0.7) -> float:
        a, b = 1 + k * p, 1 + k * (1 - p)
        return float(np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))))

    xl = ["E\nflat", *[f"{k}\n±{sd_pa(k):.2f}".replace("0.", ".") for k in KAPPAS], "K\nknown"]
    for i, design in enumerate(("XG2", "R1t")):
        for j, g in enumerate(("gain_BtoA", "gain_all12")):
            ax = axes[i, j]
            for sz in SIZES:
                for pa, ls in ((0.7, "-"), (0.5, "--")):
                    if sz == "S1" and pa != 0.7:
                        continue
                    p_true = 1.0 if sz == "S1" else pa
                    d = df[(df.design == design) & (df["size"] == sz) & np.isclose(df.pA, p_true)]
                    med = d.groupby("kappa")[g].median()
                    y = [med.loc[0.0], *[med.loc[float(k)] for k in KAPPAS], med.loc[np.inf]]
                    ax.plot(xs, y, ls, marker="o", ms=3, color=cols[sz],
                            label=f"{sz}, p_A {p_true:g}")  # fmt: skip
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(xs, xl, fontsize=7)
            ax.minorticks_off()
            ax.axhline(1, color="0.6", lw=0.7)
            where = "XG2 endpoints 4/6/8/10" if design == "XG2" else "R1 time points 1/2/10"
            what = "var(B→A)" if g == "gain_BtoA" else "total variance, all 12"
            ax.set_title(f"{where}: {what}", fontsize=10)
            if i == 1:
                ax.set_xlabel("prior strength κ / prior sd of p_A (at p_A = 0.7)")
            if j == 0:
                ax.set_ylabel("gain vs single known A founders")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Lab knowledge on the founder distribution as a prior "
                 "(median over the 9 truths)", fontsize=11)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "lab_prior.png", dpi=150)
    plt.close(fig)


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    df = table()
    df.to_csv(TAB / "lab_prior_fisher.csv", index=False)
    figure(df)
    summ = df.groupby(["design", "dist", "prior"], sort=False)[
        ["gain_BtoA", "gain_all12", "recovered_BtoA", "recovered_all12"]].median().reset_index()
    summ.to_csv(TAB / "lab_prior_fisher_summary.csv", index=False)
    pd.set_option("display.width", 200)
    show = ["S1_pA1.0", "S2_pA0.7", "S2_pA0.5", "S3_pA0.7", "S3_pA0.5", "S4_pA0.7", "S4_pA0.5"]
    for d in ("XG2", "R1t"):
        s = summ[summ.design == d]
        for v, c in (("gain_BtoA", ["E", "lab10", "lab20", "lab100", "lab1000", "K"]),
                     ("gain_all12", ["E", "lab10", "lab20", "lab100", "lab1000", "K"]),
                     ("recovered_BtoA", ["lab10", "lab20", "lab100", "lab1000"]),
                     ("recovered_all12", ["lab10", "lab20", "lab100", "lab1000"])):
            print(d, v)
            print(s.pivot(index="dist", columns="prior", values=v).loc[show, c].round(2).to_string())


if __name__ == "__main__":
    main()
