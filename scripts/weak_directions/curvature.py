"""Straight sloppy direction or curved ridge? Likelihood along +-t v at n = 512.

For one dataset per truth (n = 512, seed 0) and the two weakest Fisher eigenvectors v
(at the truth), the exact log likelihood is evaluated around the TRUTH (at n = 512 almost
every MLE sits on a face of the simplex, so it is not a usable centre):

* along the straight line  p(t) = p_truth + t v                   ("line"),
* profiled over the 9 orthogonal tangent directions at each t      ("profile"),

and compared with the quadratic (Fisher) prediction -0.5 t^2 n 42 lam. The truth is not
the maximum (the score is O(sqrt n)), so the summary uses the even part
(f(t) + f(-t)) / 2 - f(0), which removes the linear (score) term. For an eigenvector
the line and the profile coincide in the Gaussian approximation; a profile much flatter
than the line means the ridge bends away from the straight direction (curved), a profile
that tracks the quadratic means a straight sloppy direction. The profile is maximised by
SLSQP with the exact gradient along our own tangent directions (``log_likelihood_P``) and
the constraints p >= 1e-9.

  uv run python scripts/weak_directions/curvature.py
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "2")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import FIG, NAMES, OUT, PRIOR_PREC, TAB, TRUTHS, XG2, row_contrasts  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

N = 512
SEED = 0
TS = (0, -3, -2, -1, -0.5, 0.5, 1, 2, 3)
LO = 1e-9


def main() -> None:
    from exact_gw_mle import _setup

    U = row_contrasts()
    rows = np.array([0] * 5 + [1] * 5)
    dP = np.array([U[6 * r : 6 * r + 6, d] for d, r in enumerate(rows)])
    z = np.load(OUT / "fisher_eigen.npz")
    out = []
    for gt in TRUTHS:
        run = XG2 / gt / f"n{N}_s{SEED}"
        _, _, lik, names, _, _ = _setup(run)
        assert names[:12] == NAMES
        tr = json.loads((run / "mle.json").read_text())["truth"]
        p0 = np.array([tr[k] for k in NAMES])

        def ll(c, grad=True):
            p = p0 + U @ c
            if np.any(p < -1e-12):
                return -np.inf, np.zeros(10)
            v, g = lik.log_likelihood_P(np.clip(p, 0, None).reshape(2, 6), dP if grad else
                                        np.zeros((0, 6)), rows if grad else
                                        np.zeros(0, np.int64), grad)  # fmt: skip
            return v, g

        l0, _ = ll(np.zeros(10), grad=False)
        t_start = time.time()
        for k in (0, 1):
            v = z[f"{gt}_V"][:, k]
            lam = z[f"{gt}_lam"][k]
            cv = U.T @ v
            Q = np.linalg.svd(cv[None, :])[2][1:].T  # 10 x 9 complement
            sd = 1 / np.sqrt(PRIOR_PREC * (1 + N * lam))
            w = np.zeros(9)
            w0 = np.zeros(9)
            for sgn in (0, 1, -1):
                w = np.zeros(9) if sgn == 0 else w0
                for tt in sorted([t for t in TS if np.sign(t) == sgn], key=abs):
                    t = tt * sd
                    lline = ll(t * cv, grad=False)[0]

                    def f(wv, t=t):
                        val, g = ll(t * cv + Q @ wv)
                        if not np.isfinite(val):
                            return 1e10, np.zeros(9)
                        return -(val - l0), -(Q.T @ g)

                    cons = {"type": "ineq", "fun": lambda wv, t=t: p0 + U @ (t * cv + Q @ wv)
                            - LO, "jac": lambda wv: U @ Q}  # fmt: skip
                    best = None
                    for w_start in (w, np.zeros(9)):
                        # start must be feasible: shrink towards the line if needed
                        ws = w_start.copy()
                        for _ in range(30):
                            if np.all(p0 + U @ (t * cv + Q @ ws) >= LO):
                                break
                            ws *= 0.5
                        r = minimize(f, ws, jac=True, method="SLSQP", constraints=[cons],
                                     options={"maxiter": 200, "ftol": 1e-9})  # fmt: skip
                        if best is None or r.fun < best.fun:
                            best = r
                    w = best.x
                    if sgn == 0:
                        w0 = w.copy()
                    feas_line = bool(np.all(p0 + U @ (t * cv) >= 0))
                    out.append({
                        "truth": gt, "k": k + 1, "t_sd": tt, "t": t, "lam": lam,
                        "line": (lline - l0) if feas_line and np.isfinite(lline) else np.nan,
                        "profile": -best.fun, "quad": -0.5 * t**2 * N * PRIOR_PREC * lam,
                        "shift_norm": float(np.linalg.norm(Q @ w)),
                        "converged": bool(best.success),
                    })  # fmt: skip
                    print(f"{gt} w{k + 1} t={tt:+.1f}sd line {out[-1]['line']:8.2f} "
                          f"prof {out[-1]['profile']:8.2f} quad {out[-1]['quad']:8.2f} "
                          f"shift {out[-1]['shift_norm']:.3f}", flush=True)  # fmt: skip
        print(f"{gt}: {time.time() - t_start:.0f} s", flush=True)
        pd.DataFrame(out).to_csv(OUT / "curvature.csv", index=False)
    df = pd.DataFrame(out)
    df.to_csv(TAB / "curvature.csv", index=False)
    summary(df)


def summary(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from analyse_fisher import TRUTH_SHORT, _style

    rows = []
    for (gt, k), g in df.groupby(["truth", "k"], sort=False):
        g = g.set_index("t_sd")
        p0 = g.loc[0, "profile"]
        r = {"truth": gt, "k": k, "lam": g.lam.iloc[0]}
        for a in (1, 2, 3):
            r[f"quad_even_{a}sd"] = g.loc[a, "quad"]
            r[f"line_even_{a}sd"] = 0.5 * (g.loc[a, "line"] + g.loc[-a, "line"])
            r[f"profile_even_{a}sd"] = 0.5 * (g.loc[a, "profile"] + g.loc[-a, "profile"]) - p0
        r["line_odd_1sd"] = 0.5 * (g.loc[1, "line"] - g.loc[-1, "line"])
        r["profile_over_line_2sd"] = r["profile_even_2sd"] / r["line_even_2sd"]
        r["line_over_quad_2sd"] = r["line_even_2sd"] / r["quad_even_2sd"]
        r["max_shift_3sd"] = float(g.loc[[-3, 3], "shift_norm"].max())
        rows.append(r)
    s = pd.DataFrame(rows)
    s.to_csv(TAB / "curvature_summary.csv", index=False)
    (TAB / "curvature_summary.md").write_text(s.to_markdown(index=False, floatfmt=".2f") + "\n")
    _style()
    fig, axes = plt.subplots(2, 9, figsize=(7.4, 3.4), sharey=True)
    for j, gt in enumerate(TRUTHS):
        for k in (1, 2):
            ax = axes[k - 1, j]
            g = df[(df.truth == gt) & (df.k == k)].sort_values("t_sd")
            p0 = g[g.t_sd == 0].profile.iloc[0]
            sc = g.set_index("t_sd")
            s1 = 0.5 * (sc.loc[1, "line"] - sc.loc[-1, "line"])
            for col, c, ls in (("quad", "#c3c2b7", "-"), ("line", "#2a78d6", "-"),
                               ("profile", "#eb6834", "--")):  # fmt: skip
                y = g[col] - (p0 if col == "profile" else 0)
                if col == "quad" and np.isfinite(s1):  # Fisher parabola + the line's score
                    y = y + s1 * g.t_sd
                ax.plot(g.t_sd, y, ls, color=c, lw=1.3, label=col)
            ax.set_ylim(-12, 1)
            ax.set_xticks([-3, 0, 3])
            ax.tick_params(labelsize=6)
            if k == 1:
                ax.set_title(TRUTH_SHORT[gt], fontsize=8)
            if j == 0:
                ax.set_ylabel(f"Δ log lik, w{k}", fontsize=7)
    fig.supxlabel("t along the eigenvector (in predicted posterior sd at n = 512)", fontsize=7.5)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, ["quadratic (Fisher)", "straight line", "profile (9 others free)"],
               loc="upper center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 1.0))  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "curvature.png")
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summary(pd.read_csv(TAB / "curvature.csv"))
    else:
        main()
