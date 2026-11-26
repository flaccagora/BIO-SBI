"""Optimal designs for the exact two-type GW model: per truth, robust, practical protocols.

Reads the cached information (``compute_info.py``) and writes ``results/exact_gw/
optimal_design/designs.npz`` plus the CSV/markdown tables of ``docs/optimal_design/tables``:

* ``optimal_designs``: per truth x criterion (E, c = B->A, D) x design space x cost model;
* ``robust_designs``: maximin / mean efficiency over the 9 truths (E, c, D, and joint E+c
  maximin by bisection), per design space and cost model;
* ``prior_robust``: robust designs over 60 flat-prior draws (XG2 plausibility filter;
  generations 1..10), cross-evaluated with the 9-truth designs;
* ``rounded``: the best simple designs (<= 4 design points, shares in eighths) by joint
  E+c maximin efficiency over the 9 truths;
* ``protocols``: the practical protocols x truths (lam_min, 1/lam_min, efficiencies, B->A
  sd at N = 512 / 2048, clones and founders for B->A sd <= 0.05, detection fraction);
* ``face``: T1/T2/T4 at the exact boundary on the active face ("B never divides" known).

  uv run --with cvxpy python scripts/optimal_design/optimise.py
"""

from __future__ import annotations

import itertools
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from design_lib import (  # noqa: E402
    criteria, cvar, efficiencies, equivalence_gap, info, lam_min, optimal_values, optimise,
    robust,
)  # fmt: skip
from od_common import (  # noqa: E402
    BOUNDARY, CACHE, OUT, POINTS, PRIOR_PREC, SHORTNAME, TAB, TRUTHS, c_vector,
)  # fmt: skip

warnings.filterwarnings("ignore", message=".*Solution may be inaccurate.*")
warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")

LAB = [f"{f}{h}" for f, h in POINTS]
GEN = np.array([h for _, h in POINTS])
ISA = np.array([f == "A" for f, _ in POINTS])
SPACES = {
    "full": np.ones(len(POINTS), bool),
    "h>=2": GEN >= 2,
    "h>=3": GEN >= 3,
    "h>=4": GEN >= 4,
    "A only": ISA,
    "A only, h>=2": ISA & (GEN >= 2),
    "A only, h>=4": ISA & (GEN >= 4),
}
COSTS = ("clone", "founder", "founder+time")
NS = (512, 2048)
SD_TARGET = 0.05
ROUND_CANDIDATES = ["A1", "A2", "A3", "A4", "A6", "A8", "A10", "A12", "B1", "B2", "B3", "B4"]


def W(**shares) -> np.ndarray:
    w = np.zeros(len(POINTS))
    for k, v in shares.items():
        w[LAB.index(k)] = v
    return w / w.sum()


PROTOCOLS = {
    "current: A 4/6/8/10": W(A4=1, A6=1, A8=1, A10=1),
    "A 1/2/3/4": W(A1=1, A2=1, A3=1, A4=1),
    "A+B 1/2/3/4 (50/50)": W(A1=1, A2=1, A3=1, A4=1, B1=1, B2=1, B3=1, B4=1),
    "A 4/6/8/10 + B 2/4": W(A4=1, A6=1, A8=1, A10=1, B2=1, B4=1),
    "A+B 4/6/8/10 (50/50)": W(A4=1, A6=1, A8=1, A10=1, B4=1, B6=1, B8=1, B10=1),
    # rounded robust designs (rounded search, section 4: best or within 0.5% of the best
    # with A10, already an endpoint of the current protocol, in place of A12)
    "R1 robust: 3/8 A2 + 1/4 B1 + 1/4 B2 + 1/8 A10": W(A2=3, B1=2, B2=2, A10=1),
    "R2 robust h>=2: 1/4 A2 + 5/8 B2 + 1/8 A10": W(A2=2, B2=5, A10=1),
    "R3 robust A only: 1/8 A1 + 3/8 A3 + 3/8 A4 + 1/8 A6": W(A1=1, A3=3, A4=3, A6=1),
}


def cost_vector(q: np.ndarray, model: str) -> np.ndarray:
    """Cost of one detected clone per design point (``q``: P(detected), (2, H) flat)."""
    q = q.reshape(-1)
    if model == "clone":
        return np.ones_like(q)
    if model == "founder":
        return 1.0 / q
    if model == "founder+time":  # half fixed per founder, half culture time; gen 4 = 1
        return (1.0 / q) * (1.0 + GEN / 4.0) / 2.0
    raise ValueError(model)


def load(label: str) -> dict:
    z = np.load(CACHE / f"{label}.npz")
    k = z["K"].shape[-1]
    return {"K": z["K"].reshape(-1, k, k), "q": z["q"].reshape(-1), "size": z["size"].reshape(-1),
            "c": c_vector(z["U"]), "U": z["U"], "gens": z["gens"]}  # fmt: skip


def support(w: np.ndarray, tol: float = 0.005) -> str:
    return " ".join(f"{LAB[i]}:{w[i]:.2f}" for i in np.flatnonzero(w > tol))


def sd(cv: float, n: float) -> float:
    """B->A asymptotic sd for n clones from c^T K^+ c per clone (whitened)."""
    return float(np.sqrt(cv / PRIOR_PREC / n))


def sd_bayes(K: np.ndarray, c: np.ndarray, n: float) -> float:
    k = K.shape[0]
    return float(np.sqrt(c @ np.linalg.solve(np.eye(k) + n * K, c) / PRIOR_PREC))


def n_for_sd(K: np.ndarray, c: np.ndarray, target: float = SD_TARGET,
             prior: bool = False) -> float:  # fmt: skip
    if not prior:
        return float(cvar(K, c) / PRIOR_PREC / target**2)
    lo, hi = 1.0, 1e8
    if sd_bayes(K, c, hi) > target:
        return float("inf")
    if sd_bayes(K, c, lo) <= target:
        return 0.0
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if sd_bayes(K, c, mid) > target else (lo, mid)
    return float(hi)


def joint_maximin(Kset, cset, opts, mask=None, tol=1e-4) -> tuple[np.ndarray, float]:
    """Maximise min over truths of min(E-eff, c-eff): bisection on the level (convex
    feasibility problem per level)."""
    import cvxpy as cp

    n = len(Kset[0])
    k = Kset[0].shape[1]
    lo, hi, best = 0.0, 1.0, None
    while hi - lo > tol:
        tau = 0.5 * (lo + hi)
        w = cp.Variable(n, nonneg=True)
        cons = [cp.sum(w) == 1]
        if mask is not None:
            cons.append(w[~mask] == 0)
        for Ks, c, o in zip(Kset, cset, opts, strict=True):
            s = 1.0 / o["E"]["lam_min"]
            M = sum(w[i] * (Ks[i] * s) for i in range(n))
            M = 0.5 * (M + M.T)
            cons.append(M - tau * np.eye(k) >> 0)
            cons.append(cp.matrix_frac(c, M) <= o["c"]["cvar"] / s / tau)
        prob = cp.Problem(cp.Minimize(0), cons)
        try:
            prob.solve(solver="CLARABEL")
            ok = prob.status in ("optimal", "optimal_inaccurate")
        except cp.error.SolverError:
            ok = False
        if ok:
            lo, best = tau, np.clip(w.value, 0, None)
        else:
            hi = tau
    best[best < 1e-6 * best.max()] = 0
    return best / best.sum(), lo


def scaled(d: dict, cost: str) -> np.ndarray:
    return d["K"] / cost_vector(d["q"], cost)[:, None, None]


def eff_rows(name, w, Kset, cset, opts, truths, extra=None) -> list[dict]:
    rows = []
    for gt, Ks, c, o in zip(truths, Kset, cset, opts, strict=True):
        e = efficiencies(Ks, c, w, o)
        rows.append({"design": name, "truth": SHORTNAME.get(gt, gt), **(extra or {}),
                     "eff_E": e["eff_E"], "eff_c": e["eff_c"], "eff_D": e["eff_D"]})  # fmt: skip
    return rows


def rounded_search(Kset, cset, opts, allowed: list[str], max_points: int = 4,
                   parts: int = 8) -> pd.DataFrame:  # fmt: skip
    """All designs on <= max_points of ``allowed`` with shares in 1/parts; score =
    min over truths of min(E-eff, c-eff)."""
    idx = [LAB.index(a) for a in allowed]
    designs = []
    for k in range(1, max_points + 1):
        comps = [c for c in itertools.product(range(1, parts + 1), repeat=k) if sum(c) == parts]
        for sub in itertools.combinations(idx, k):
            for comp in comps:
                w = np.zeros(len(POINTS))
                w[list(sub)] = np.array(comp) / parts
                designs.append(w)
    Wm = np.array(designs)  # (n_designs, 24)
    effE = np.empty((len(Wm), len(Kset)))
    effc = np.empty_like(effE)
    for t, (Ks, c, o) in enumerate(zip(Kset, cset, opts, strict=True)):
        M = np.einsum("dp,pij->dij", Wm, Ks)
        lam, V = np.linalg.eigh(M)
        effE[:, t] = lam[:, 0] / o["E"]["lam_min"]
        a = np.einsum("dij,i->dj", V, c)
        tol = 1e-10 * lam[:, -1:]
        with np.errstate(divide="ignore"):
            v = np.where(lam > tol, a**2 / np.where(lam > tol, lam, 1.0),
                         np.where(np.abs(a) > 1e-8, np.inf, 0.0)).sum(1)  # fmt: skip
        effc[:, t] = o["c"]["cvar"] / v
    score = np.minimum(effE, effc).min(1)
    order = np.argsort(-score)[:15]
    return pd.DataFrame({"design": [support(Wm[i], 0) for i in order], "score": score[order],
                         "min_eff_E": effE[order].min(1), "min_eff_c": effc[order].min(1),
                         "w": [Wm[i] for i in order]})  # fmt: skip


def write(df: pd.DataFrame, name: str, fmt: str = ".3g") -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False)
    with open(TAB / f"{name}.md", "w") as f:
        f.write(df.to_markdown(index=False, floatfmt=fmt))


def main() -> None:
    data = {gt: load(gt) for gt in TRUTHS}
    res: dict = {"points": LAB}

    # ---------------------------------------------------------------- 1. per truth
    opt = {}  # (cost, space) -> list of opts per truth
    rows = []
    for cost in COSTS:
        for space, mask in SPACES.items():
            if cost != "clone" and space not in ("full", "h>=2"):
                continue
            ol = []
            for gt in TRUTHS:
                d = data[gt]
                Ks = scaled(d, cost)
                o = {}
                for crit in ("E", "c", "D"):
                    w = optimise(Ks, d["c"], crit, mask=mask)
                    o[crit] = {"w": w, **criteria(Ks, d["c"], w)}
                    clone_share = (w / cost_vector(d["q"], cost))
                    clone_share /= clone_share.sum()
                    rows.append({
                        "truth": SHORTNAME[gt], "space": space, "cost": cost,
                        "criterion": crit, "design (budget shares)": support(w),
                        "design (clone shares)": support(clone_share),
                        "lam_min": o[crit]["lam_min"], "1/lam_min": 1 / o[crit]["lam_min"],
                        "sd_BtoA_512": sd(o[crit]["cvar"], 512),
                        "eq_gap": equivalence_gap(Ks, d["c"], w, crit),
                    })  # fmt: skip
                ol.append(o)
            opt[(cost, space)] = ol
    write(pd.DataFrame(rows), "optimal_designs")
    res["opt_full"] = {gt: {k: v["w"] for k, v in o.items()}
                       for gt, o in zip(TRUTHS, opt[("clone", "full")], strict=True)}  # fmt: skip

    # ---------------------------------------------------------------- 2. robust (9 truths)
    rrows, effrows, rob = [], [], {}
    for cost in COSTS:
        Kset = [scaled(data[gt], cost) for gt in TRUTHS]
        cset = [data[gt]["c"] for gt in TRUTHS]
        for space, mask in SPACES.items():
            if cost != "clone" and space not in ("full", "h>=2"):
                continue
            opts_full = opt[(cost, "full")]  # efficiencies relative to the unrestricted optimum
            designs = {}
            for crit in ("E", "c", "D"):
                for mode in ("maximin", "mean"):
                    designs[f"{crit} {mode}"] = robust(Kset, cset, opts_full, crit, mode, mask)
            designs["E+c maximin"], _ = joint_maximin(Kset, cset, opts_full, mask)
            for name, w in designs.items():
                rob[(cost, space, name)] = w
                er = eff_rows(name, w, Kset, cset, opts_full, TRUTHS, {"space": space, "cost": cost})
                effrows += er
                e = pd.DataFrame(er)
                rrows.append({"space": space, "cost": cost, "robust design": name,
                              "budget shares": support(w),
                              "min eff_E": e.eff_E.min(), "median eff_E": e.eff_E.median(),
                              "min eff_c": e.eff_c.min(), "median eff_c": e.eff_c.median(),
                              "min eff_D": e.eff_D.min(), "median eff_D": e.eff_D.median()})  # fmt: skip
    write(pd.DataFrame(rrows), "robust_designs")
    write(pd.DataFrame(effrows), "robust_efficiencies")
    res["robust"] = {" | ".join(k): v for k, v in rob.items()}

    # ---------------------------------------------------------------- 3. prior draws
    draws = np.load(OUT / "prior_draws.npz")["p12"]
    pl = [f"prior{i:03d}" for i in range(len(draws))]
    sub = np.array([h <= 10 for _, h in POINTS])  # prior caches: generations 1..10
    pdat = []
    for lab in pl:
        d = load(lab)
        K = np.zeros((len(POINTS),) + d["K"].shape[1:])
        K[sub] = d["K"]
        pdat.append({"K": K, "c": d["c"], "q": d["q"]})
    popts = []
    for d in pdat:
        o = {}
        for crit in ("E", "c", "D"):
            w = optimise(d["K"], d["c"], crit, mask=sub)
            o[crit] = {"w": w, **criteria(d["K"], d["c"], w)}
        popts.append(o)
    Kp, cp_ = [d["K"] for d in pdat], [d["c"] for d in pdat]
    Kt, ct = [data[gt]["K"] for gt in TRUTHS], [data[gt]["c"] for gt in TRUTHS]
    # 9-truth optima restricted to gens <= 10 (for a fair cross-evaluation)
    topts10 = []
    for Ks, c in zip(Kt, ct, strict=True):
        o = {}
        for crit in ("E", "c", "D"):
            w = optimise(Ks, c, crit, mask=sub)
            o[crit] = {"w": w, **criteria(Ks, c, w)}
        topts10.append(o)
    pdesigns = {
        "prior: E mean": robust(Kp, cp_, popts, "E", "mean", sub),
        "prior: c mean": robust(Kp, cp_, popts, "c", "mean", sub),
        "prior: E maximin": robust(Kp, cp_, popts, "E", "maximin", sub),
        "prior: c maximin": robust(Kp, cp_, popts, "c", "maximin", sub),
        "prior: E+c maximin": joint_maximin(Kp, cp_, popts, sub)[0],
        "9 truths: E+c maximin (gens<=10)": joint_maximin(Kt, ct, topts10, sub)[0],
        "current: A 4/6/8/10": PROTOCOLS["current: A 4/6/8/10"],
    }
    prow, pefrows = [], []
    for name, w in pdesigns.items():
        ep = pd.DataFrame(eff_rows(name, w, Kp, cp_, popts, pl))
        et = pd.DataFrame(eff_rows(name, w, Kt, ct, topts10, TRUTHS))
        pefrows += ep.assign(set="prior").to_dict("records") + et.assign(set="9 truths").to_dict("records")
        prow.append({"design": name, "budget shares": support(w),
                     "prior: median eff_E": ep.eff_E.median(), "prior: 10% eff_E": ep.eff_E.quantile(0.1),
                     "prior: min eff_E": ep.eff_E.min(),
                     "prior: median eff_c": ep.eff_c.median(), "prior: 10% eff_c": ep.eff_c.quantile(0.1),
                     "prior: min eff_c": ep.eff_c.min(),
                     "9 truths: min eff_E": et.eff_E.min(), "9 truths: min eff_c": et.eff_c.min()})  # fmt: skip
        res[f"prior_design | {name}"] = w
    write(pd.DataFrame(prow), "prior_robust")
    write(pd.DataFrame(pefrows), "prior_efficiencies")
    psup = pd.DataFrame([{"draw": lab, **{f"{cr}: {LAB[i]}": o[cr]["w"][i] for cr in "Ec"
                          for i in range(len(POINTS)) if sub[i]}} for lab, o in zip(pl, popts)])  # fmt: skip
    psup.to_csv(TAB / "prior_optimal_weights.csv", index=False)

    # ---------------------------------------------------------------- 4. rounded designs
    opts_full = opt[("clone", "full")]
    rd = {}
    rrw = []
    for space, allowed in [("full", ROUND_CANDIDATES),
                           ("h>=2", [a for a in ROUND_CANDIDATES if int(a[1:]) >= 2]),
                           ("A only", [a for a in ROUND_CANDIDATES if a[0] == "A"])]:  # fmt: skip
        df = rounded_search(Kt, ct, opts_full, allowed)
        rd[space] = df.w.iloc[0]
        rrw += df.drop(columns="w").head(8).assign(space=space).to_dict("records")
    write(pd.DataFrame(rrw)[["space", "design", "score", "min_eff_E", "min_eff_c"]], "rounded")
    protocols = dict(PROTOCOLS)
    res["protocols"] = protocols
    res["rounded_best"] = rd

    # ---------------------------------------------------------------- 5. protocols table
    cur = protocols["current: A 4/6/8/10"]
    prow = []
    curves = {}
    Ngrid = np.unique(np.round(np.logspace(np.log10(64), np.log10(32768), 60)))
    for name, w in protocols.items():
        for gt, Ks, c, o in zip(TRUTHS, Kt, ct, opts_full, strict=True):
            d = data[gt]
            M = info(Ks, w)
            Mc = info(Ks, cur)
            e = efficiencies(Ks, c, w, o)
            det_frac = 1.0 / np.sum(w / d["q"])
            n05 = n_for_sd(M, c)
            n05b = n_for_sd(M, c, prior=True)
            prow.append({
                "protocol": name, "design": support(w, 0), "truth": SHORTNAME[gt],
                "lam_min": e["lam_min"], "1/lam_min (clones)": 1 / e["lam_min"],
                "eff_E": e["eff_E"], "eff_c": e["eff_c"], "eff_D": e["eff_D"],
                "gain lam_min vs current": e["lam_min"] / lam_min(Mc),
                "gain B->A var vs current": cvar(Mc, c) / e["cvar"],
                "sd_BtoA N=512": sd(e["cvar"], 512), "sd_BtoA N=2048": sd(e["cvar"], 2048),
                "sd_BtoA N=512 (prior)": sd_bayes(M, c, 512),
                "sd_BtoA N=2048 (prior)": sd_bayes(M, c, 2048),
                "N for sd<=0.05": n05, "N for sd<=0.05 (prior)": n05b,
                "detected fraction": det_frac, "founders for sd<=0.05": n05 / det_frac,
                "mean cells per detected clone": float(np.sum(w * d["size"])),
            })  # fmt: skip
            curves[(name, gt)] = {
                "sd": [sd(e["cvar"], n) for n in Ngrid],
                "sd_prior": [sd_bayes(M, c, n) for n in Ngrid],
                "inv_lam_ratio": [1 / (n * e["lam_min"]) for n in Ngrid],
            }
    prot = pd.DataFrame(prow)
    write(prot, "protocols")
    # compact summary: per protocol, range over truths
    summ = prot.groupby("protocol", sort=False).agg(
        design=("design", "first"),
        min_eff_E=("eff_E", "min"), median_eff_E=("eff_E", "median"),
        min_eff_c=("eff_c", "min"), median_eff_c=("eff_c", "median"),
        gain_lam_min=("gain lam_min vs current", "median"),
        gain_lam_min_lo=("gain lam_min vs current", "min"),
        gain_lam_min_hi=("gain lam_min vs current", "max"),
        gain_BtoA_var=("gain B->A var vs current", "median"),
        gain_BtoA_var_lo=("gain B->A var vs current", "min"),
        gain_BtoA_var_hi=("gain B->A var vs current", "max"),
        detected_fraction_min=("detected fraction", "min"),
    ).reset_index()
    write(summ, "protocols_summary")
    for col in ("1/lam_min (clones)", "sd_BtoA N=512", "sd_BtoA N=2048", "N for sd<=0.05",
                "founders for sd<=0.05", "eff_E", "eff_c"):  # fmt: skip
        pv = prot.pivot(index="protocol", columns="truth", values=col)
        pv = pv.loc[list(protocols), [SHORTNAME[t] for t in TRUTHS]].reset_index()
        safe = col.replace("/", "_").replace("<=", "le").replace(" ", "_").replace("(", "").replace(")", "")
        write(pv, f"protocols_{safe}")
    res["curves_N"] = Ngrid
    res["curves"] = {" | ".join(k): v for k, v in curves.items()}

    # ---------------------------------------------------------------- 6. boundary face
    frows = []
    for gt in BOUNDARY:
        d = load(f"{gt}_face")
        o = optimal_values(d["K"], d["c"])
        de = load(gt)  # eps variant, full space
        for name, w in list(protocols.items()) + [("face E-optimal", o["E"]["w"]),
                                                  ("face c-optimal", o["c"]["w"])]:  # fmt: skip
            e = efficiencies(d["K"], d["c"], w, o)
            frows.append({"truth": SHORTNAME[gt], "design": name, "support": support(w),
                          "face lam_min": e["lam_min"], "face 1/lam_min": 1 / e["lam_min"],
                          "face eff_E": e["eff_E"], "face eff_c": e["eff_c"],
                          "face sd_BtoA N=512": sd(e["cvar"], 512),
                          "eps0.01 sd_BtoA N=512": sd(cvar(info(de["K"], w), de["c"]), 512)})  # fmt: skip
    write(pd.DataFrame(frows), "face")

    # A-founded exact-boundary vs eps (check of the eps device on A-founded designs)
    chk = []
    for gt in BOUNDARY:
        ex, ep = load(f"{gt}_exact"), load(gt)
        for name in ("current: A 4/6/8/10", "A 1/2/3/4"):
            w = protocols[name]
            chk.append({"truth": SHORTNAME[gt], "design": name,
                        "1/lam exact": 1 / lam_min(info(ex["K"], w)),
                        "1/lam eps0.01": 1 / lam_min(info(ep["K"], w)),
                        "sd512 exact": sd(cvar(info(ex["K"], w), ex["c"]), 512),
                        "sd512 eps0.01": sd(cvar(info(ep["K"], w), ep["c"]), 512)})  # fmt: skip
    write(pd.DataFrame(chk), "boundary_eps_check")

    # detection per point
    det = pd.DataFrame([{"truth": SHORTNAME[gt], **{LAB[i]: data[gt]["q"][i] for i in range(len(POINTS))}}
                        for gt in TRUTHS])  # fmt: skip
    write(det, "detection")
    size = pd.DataFrame([{"truth": SHORTNAME[gt], **{LAB[i]: data[gt]["size"][i] for i in range(len(POINTS))}}
                         for gt in TRUTHS])  # fmt: skip
    write(size, "clone_size", ".1f")

    np.savez(OUT / "designs.npz", payload=np.array(json.dumps(res, default=lambda x: np.asarray(x).tolist())))
    print("done")


if __name__ == "__main__":
    main()
