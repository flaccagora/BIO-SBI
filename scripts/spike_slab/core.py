"""Spike-and-slab on B's divisions for the exact two-type GW (XG2) runs: shared code.

Models ("faces") are the full model with some B entries pinned at 0; every row keeps a
flat Dirichlet(1) on its remaining outcomes. Coordinates are logit sticks ``u`` (same
stick order as the NUTS runs: the row's entry order with the pinned entries dropped).
Under Dir(1) on ``m`` outcomes the sticks are independent Beta(1, m-1-j), so

    log target(u) = loglik(P(u)) + sum_j [log b_j + log expit(u_j) + b_j log expit(-u_j)]

is a *normalised* prior times the likelihood: its integral over ``u`` is the evidence.

Evidence estimators (``evidence``):
* Laplace in ``u`` (mode of the target, finite-difference Hessian of the exact gradient);
* importance sampling with a two-stage multivariate-t mixture proposal (balance
  heuristic over the stages), diagnosed by ESS and the Pareto k of the weights;
* bridge sampling (Meng & Wong, iterative) between the NUTS draws and the same proposal
  (full model only, where NUTS draws exist).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.special import gammaln, log_expit, logsumexp  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
DATA = Path("/home/matteo/BIO-SBI/results/exact_gw/data_size")
OUT = Path("/home/matteo/BIO-SBI/results/exact_gw/spike_slab")
DOCS = REPO / "docs" / "spike_slab"

DIV = ("B.div_AA", "B.div_AB", "B.div_BB")
#: model name -> B entries pinned at 0
MODELS: dict[str, tuple[str, ...]] = {
    "full": (),
    "M0": DIV,
    "nodediff": ("B.to_A", "B.div_AA", "B.div_AB"),
    "z_AA": ("B.div_AA",),
    "z_AB": ("B.div_AB",),
    "z_BB": ("B.div_BB",),
    "z_AA_AB": ("B.div_AA", "B.div_AB"),
    "z_AA_BB": ("B.div_AA", "B.div_BB"),
    "z_AB_BB": ("B.div_AB", "B.div_BB"),
}
FACES8 = ["full", "z_AA", "z_AB", "z_BB", "z_AA_AB", "z_AA_BB", "z_AB_BB", "M0"]
PRIMARY = ["full", "M0", "nodediff"]
FREE12 = ["A.death", "A.persist", "A.to_B", "A.div_AA", "A.div_AB", "A.div_BB",
          "B.death", "B.to_A", "B.persist", "B.div_AA", "B.div_AB", "B.div_BB"]  # fmt: skip


# ---------------------------------------------------------------------------- runs
def list_runs(require_post: bool = True) -> list[tuple[str, int, int, Path]]:
    """(gt, n, seed, path) of every finished run with ``mle.json`` (sorted)."""
    out = []
    for m in sorted(DATA.glob("*/n*_s*/mle.json")):
        run = m.parent
        k = re.fullmatch(r"n(\d+)_s(\d+)", run.name)
        if k is None or (require_post and not (run / "posterior_samples.csv.gz").exists()):
            continue
        out.append((run.parent.name, int(k.group(1)), int(k.group(2)), run))
    return sorted(out, key=lambda r: (r[0], r[1], r[2]))


def run_dir(gt: str, n: int, seed: int) -> Path:
    return OUT / "runs" / gt / f"n{n}_s{seed}"


def load(run: Path, kernel: str = "adjoint"):
    """``(lik, names, truth, post, mle)`` of a run (exact likelihood, ``adjoint`` kernel)."""
    from exact_gw_mle import _setup

    _cfg, _sim, lik, names, truth, post = _setup(run, extra=(f"inference.method.kernel={kernel}",))
    mle = json.loads((run / "mle.json").read_text())
    return lik, names, truth, post, mle


# ---------------------------------------------------------------------------- faces
class Face:
    """Exact likelihood restricted to a face (``zeros`` pinned at 0), flat Dirichlet rows."""

    def __init__(self, lik, names: list[str], zeros: tuple[str, ...], template: np.ndarray,
                 coords: str = "alr", ref_phys: np.ndarray | None = None):
        """``coords``: ``alr`` (log ratios to a reference outcome per row, the largest
        entry of ``ref_phys``; default) or ``u`` (logit sticks, the NUTS coordinates)."""
        self.lik, self.names, self.zeros = lik, list(names), tuple(zeros)
        self.coords = coords
        sim = lik.sim
        self.space, self.law = sim.space, sim.law
        self.template = np.asarray(template, dtype=float).copy()  # founder entries
        for z in zeros:
            self.template[self.names.index(z)] = 0.0
        self.blocks = []  # (type index k, free entry idx in row, name idx, b_j array)
        for r in self.space.rows:
            if not r.n_sticks:
                continue
            k = self.law.types.index(r.prefix)
            free = [i for i in r.free if f"{r.prefix}.{r.entries[i]}" not in zeros]
            assert r.mass == 1.0 and not r.fixed
            idx = [self.names.index(f"{r.prefix}.{r.entries[i]}") for i in free]
            m = len(free)
            self.blocks.append((k, free, idx, np.arange(m - 1, 0, -1, dtype=float)))
        self.dim = sum(len(b[1]) - 1 for b in self.blocks)
        rp = self.template if ref_phys is None else np.asarray(ref_phys)
        self.refs = [int(np.argmax(rp[idx])) for _k, _f, idx, _b in self.blocks]
        self.log_prior_const = float(sum(np.log(b[3]).sum() for b in self.blocks))
        # log prior density in the physical coordinates (Dir(1) on m outcomes: (m-1)!)
        self.log_prior_phys = float(sum(gammaln(len(b[1])) for b in self.blocks))
        self.evals = 0

    # --- coordinates
    def _sticks(self, u: np.ndarray, coords: str):
        """Per block: (p (m,), dp (m-1, m)) from ``u`` (logit) or ``s`` (stick) coords."""
        out, o = [], 0
        for bi, (_k, free, _idx, _b) in enumerate(self.blocks):
            m = len(free)
            x = u[o : o + m - 1]
            if coords == "alr":
                r = self.refs[bi]
                nr = [i for i in range(m) if i != r]
                z = np.zeros(m)
                z[nr] = x
                p = np.exp(z - logsumexp(z))
                d = -np.outer(p[nr], p)
                d[np.arange(m - 1), nr] += p[nr]
                out.append((p, d))
                o += m - 1
                continue
            if coords == "u":
                ls, l1s = log_expit(x), log_expit(-x)
            else:
                ls, l1s = np.log(x), np.log1p(-x)
            cum = np.concatenate([[0.0], np.cumsum(l1s)])  # log prod_{l<i} (1 - s_l)
            logp = np.empty(m)
            logp[:-1] = ls + cum[:-1]
            logp[-1] = cum[-1]
            p = np.exp(logp)
            d = np.zeros((m - 1, m))
            s, oms = np.exp(ls), np.exp(l1s)
            for j in range(m - 1):
                if coords == "u":  # dp_i/du_j: i == j: p_j (1 - s_j); i > j: -p_i s_j
                    d[j, j] = p[j] * oms[j]
                    d[j, j + 1 :] = -p[j + 1 :] * s[j]
                else:  # dp_i/ds_j: i == j: prod_{l<j}(1-s_l); i > j: -p_i / (1 - s_j)
                    d[j, j] = np.exp(cum[j])
                    d[j, j + 1 :] = -p[j + 1 :] / max(oms[j], 1e-300)
            out.append((p, d))
            o += m - 1
        return out

    def P_dP(self, x: np.ndarray, coords: str | None = None):
        coords = coords or self.coords
        lik = self.lik
        P = np.zeros((len(self.law.types), lik.n_outcomes_max))
        dP = np.zeros((self.dim, lik.n_outcomes_max))
        rows = np.zeros(self.dim, dtype=np.int64)
        o = 0
        for (k, free, _idx, _b), (p, d) in zip(self.blocks, self._sticks(x, coords), strict=True):
            m = len(free)
            P[k, free] = p
            dP[o : o + m - 1][:, free] = d
            rows[o : o + m - 1] = k
            o += m - 1
        return P, dP, rows

    def phys(self, x: np.ndarray, coords: str | None = None) -> np.ndarray:
        coords = coords or self.coords
        out = self.template.copy()
        for (_k, _free, idx, _b), (p, _d) in zip(self.blocks, self._sticks(x, coords), strict=True):
            out[idx] = p
        return out

    def u_of(self, phys: np.ndarray, eps: float = 1e-7, coords: str | None = None) -> np.ndarray:
        """Coordinates of a physical vector (pinned entries dropped, rows renormalised)."""
        coords = coords or self.coords
        out = []
        for bi, (_k, _free, idx, _b) in enumerate(self.blocks):
            row = np.clip(np.asarray(phys)[idx], eps, None)
            row = row / row.sum()
            if coords == "alr":
                lr = np.log(row) - np.log(row[self.refs[bi]])
                out.extend(np.delete(lr, self.refs[bi]))
                continue
            rem = np.cumsum(row[::-1])[::-1]
            s = np.clip(row[:-1] / rem[:-1], eps, 1 - eps)
            out.extend(np.log(s) - np.log1p(-s) if coords == "u" else s)
        return np.asarray(out)

    # --- densities
    def log_prior(self, u: np.ndarray) -> tuple[float, np.ndarray]:
        if self.coords == "alr":  # Dir(1) density (m-1)! times Jacobian prod_i p_i
            lp, g, o = self.log_prior_phys, [], 0
            for bi, (_k, free, _idx, _b) in enumerate(self.blocks):
                m = len(free)
                z = np.zeros(m)
                nr = [i for i in range(m) if i != self.refs[bi]]
                z[nr] = u[o : o + m - 1]
                logp = z - logsumexp(z)
                lp += float(logp.sum())
                g.extend(1 - m * np.exp(logp[nr]))
                o += m - 1
            return lp, np.asarray(g)
        b = np.concatenate([blk[3] for blk in self.blocks])
        lp = self.log_prior_const + float(np.sum(log_expit(u) + b * log_expit(-u)))
        s = np.exp(log_expit(u))
        return lp, (1 - s) - b * s

    def loglik(self, x: np.ndarray, grad: bool = True, coords: str | None = None):
        P, dP, rows = self.P_dP(x, coords)
        if not grad:
            dP, rows = dP[:0], rows[:0]
        self.evals += 1
        ll, g = self.lik.log_likelihood_P(P, dP, rows, grad=grad)
        if not np.isfinite(ll):
            return -np.inf, np.zeros(self.dim) if grad else g
        return float(ll), g

    def target(self, u: np.ndarray, grad: bool = True):
        """log likelihood + log prior density of ``u`` (normalised), gradient."""
        u = np.asarray(u, dtype=float)
        if not np.all(np.isfinite(u)):
            return -np.inf, np.zeros(self.dim)
        ll, g = self.loglik(u, grad)
        lp, gp = self.log_prior(u)
        if not np.isfinite(ll):
            return -np.inf, np.zeros(self.dim)
        return ll + lp, (g + gp) if grad else None

    def targets(self, U: np.ndarray) -> np.ndarray:
        return np.array([self.target(u, grad=False)[0] for u in U])

    # --- optimisation
    def mode(self, starts: list[np.ndarray], box: float = 30.0) -> tuple[np.ndarray, float]:
        from scipy.optimize import minimize

        def neg(u):
            v, g = self.target(u)
            if not np.isfinite(v):
                return 1e300, np.zeros_like(u)
            return -v, -g

        best = None
        for st in starts:
            r = minimize(neg, np.clip(st, -box + 1, box - 1), jac=True, method="L-BFGS-B",
                         bounds=[(-box, box)] * self.dim,
                         options={"maxiter": 2000, "ftol": 1e-13, "gtol": 1e-7})  # fmt: skip
            if best is None or r.fun < best.fun:
                best = r
        return best.x, float(-best.fun)

    def mle(self, starts_phys: list[np.ndarray]) -> tuple[np.ndarray, float]:
        """Maximum likelihood on the face (stick coordinates, bounds [1e-9, 1-1e-9])."""
        from scipy.optimize import minimize

        lo, hi = 1e-9, 1 - 1e-9

        def neg(s):
            v, g = self.loglik(np.clip(s, lo, hi), True, "s")
            if not np.isfinite(v):
                return 1e300, np.zeros_like(s)
            return -v, -g

        best = None
        for p in starts_phys:
            st = np.clip(self.u_of(p, eps=1e-6, coords="s"), 1e-6, 1 - 1e-6)
            r = minimize(neg, st, jac=True, method="L-BFGS-B", bounds=[(lo, hi)] * self.dim,
                         options={"maxiter": 1000})  # fmt: skip
            if best is None or r.fun < best.fun:
                best = r
        return self.phys(np.clip(best.x, lo, hi), "s"), float(-best.fun)

    def hessian(self, u: np.ndarray, h: float = 1e-4) -> np.ndarray:
        H = np.zeros((self.dim, self.dim))
        for j in range(self.dim):
            e = np.zeros(self.dim)
            e[j] = h
            H[:, j] = (self.target(u + e)[1] - self.target(u - e)[1]) / (2 * h)
        return -(H + H.T) / 2  # negative Hessian of the log target


# ---------------------------------------------------------------------------- t mixtures
class TMix:
    """Mixture of multivariate t distributions (weights, means, covariances, df)."""

    def __init__(self, comps: list[tuple[float, np.ndarray, np.ndarray]], df: float = 5.0):
        w = np.array([c[0] for c in comps], dtype=float)
        self.w = w / w.sum()
        self.mu = [np.asarray(c[1], dtype=float) for c in comps]
        self.L = [np.linalg.cholesky(_pd(np.asarray(c[2], dtype=float))) for c in comps]
        self.df = float(df)
        self.d = len(self.mu[0])

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        ks = rng.choice(len(self.w), size=n, p=self.w)
        z = rng.standard_normal((n, self.d))
        g = rng.chisquare(self.df, size=n) / self.df
        x = np.empty((n, self.d))
        for k in range(len(self.w)):
            sel = ks == k
            x[sel] = self.mu[k] + (z[sel] @ self.L[k].T) / np.sqrt(g[sel])[:, None]
        return x

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        d, nu = self.d, self.df
        out = []
        for w, mu, L in zip(self.w, self.mu, self.L, strict=True):
            r = np.linalg.solve(L, (x - mu).T)
            m = np.sum(r**2, axis=0)
            c = (gammaln((nu + d) / 2) - gammaln(nu / 2) - d / 2 * np.log(nu * np.pi)
                 - np.log(np.diag(L)).sum())  # fmt: skip
            out.append(np.log(w) + c - (nu + d) / 2 * np.log1p(m / nu))
        return logsumexp(np.array(out), axis=0)


def _pd(C: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    C = (C + C.T) / 2
    v, V = np.linalg.eigh(C)
    v = np.maximum(v, floor * max(v.max(), 1e-12))
    return (V * v) @ V.T


def weighted_moments(x: np.ndarray, logw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = np.exp(logw - logw.max())
    w /= w.sum()
    mu = w @ x
    xc = x - mu
    return mu, (xc * w[:, None]).T @ xc


# ---------------------------------------------------------------------------- diagnostics
def gpd_k(x: np.ndarray) -> float:
    """Zhang & Stephens (2009) estimate of the generalized Pareto shape k (arviz-style)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n < 5:
        return np.nan
    prior_bs, prior_k = 3, 10
    m_est = 30 + int(np.sqrt(n))
    b_ary = 1 - np.sqrt(m_est / (np.arange(1, m_est + 1) - 0.5))
    b_ary /= prior_bs * x[int(n / 4 + 0.5) - 1]
    b_ary += 1 / x[-1]
    k_ary = np.log1p(-b_ary[:, None] * x).mean(axis=1)
    len_scale = n * (np.log(-(b_ary / k_ary)) - k_ary - 1)
    weights = 1 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)
    real = weights >= 10 * np.finfo(float).eps
    b_ary, weights = b_ary[real], weights[real] / weights[real].sum()
    b_post = (b_ary * weights).sum()
    k_post = np.log1p(-b_post * x).mean()
    return float((n * k_post + prior_k * 0.5) / (n + prior_k))


def pareto_k(logw: np.ndarray) -> float:
    """Pareto k of the importance weights (tail of the largest 20%, min 3 sqrt(S))."""
    lw = np.asarray(logw, dtype=float)
    lw = lw[np.isfinite(lw)]
    S = len(lw)
    M = int(min(0.2 * S, 3 * np.sqrt(S)))
    lw = np.sort(lw - lw.max())
    tail = lw[-M:]
    cut = lw[-M - 1]
    return gpd_k(np.exp(tail) - np.exp(cut))


def ess_is(logw: np.ndarray) -> float:
    lw = logw[np.isfinite(logw)]
    w = np.exp(lw - lw.max())
    return float(w.sum() ** 2 / (w**2).sum())


def bridge(l1: np.ndarray, l2: np.ndarray, n1_eff: float | None = None, tol: float = 1e-10
           ) -> float:  # fmt: skip
    """Meng & Wong optimal bridge: ``l1 = log target - log q`` at posterior draws,
    ``l2`` the same at proposal draws. Returns log evidence."""
    l1, l2 = l1[np.isfinite(l1)], l2[np.isfinite(l2)]
    n1 = float(n1_eff or len(l1))
    n2 = float(len(l2))
    s1, s2 = n1 / (n1 + n2), n2 / (n1 + n2)
    lr = logsumexp(l2) - np.log(n2)  # IS start
    for _ in range(1000):
        num = logsumexp(l2 - np.logaddexp(np.log(s1) + l2, np.log(s2) + lr)) - np.log(n2)
        den = logsumexp(-np.logaddexp(np.log(s1) + l1, np.log(s2) + lr)) - np.log(len(l1))
        new = num - den
        if abs(new - lr) < tol:
            return float(new)
        lr = new
    return float(lr)


# ---------------------------------------------------------------------------- evidence
def weighted_gmm(x: np.ndarray, w: np.ndarray, k: int, rng: np.random.Generator,
                 iters: int = 60, shrink: float = 0.05) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Weighted EM for a k-component Gaussian mixture (covariances shrunk towards the
    overall weighted covariance by ``shrink``, to survive concentrated weights)."""
    w = w / w.sum()
    n, d = x.shape
    mu_all = w @ x
    C_all = ((x - mu_all) * w[:, None]).T @ (x - mu_all) + 1e-8 * np.eye(d)
    # init: weighted draws as centres (k-means++-like on the weighted sample)
    idx = rng.choice(n, size=k, replace=False, p=w) if (w > 0).sum() >= k else rng.choice(n, k)
    mus = x[idx].copy()
    Cs = [C_all.copy() for _ in range(k)]
    pis = np.full(k, 1.0 / k)
    for _ in range(iters):
        logr = np.empty((n, k))
        for j in range(k):
            L = np.linalg.cholesky(_pd(Cs[j]))
            r = np.linalg.solve(L, (x - mus[j]).T)
            logr[:, j] = np.log(pis[j]) - np.log(np.diag(L)).sum() - 0.5 * np.sum(r**2, 0)
        logr -= logsumexp(logr, axis=1, keepdims=True)
        R = np.exp(logr) * w[:, None]
        nk = R.sum(0) + 1e-300
        pis = nk / nk.sum()
        for j in range(k):
            mus[j] = R[:, j] @ x / nk[j]
            xc = x - mus[j]
            Cj = (xc * R[:, j][:, None]).T @ xc / nk[j]
            Cs[j] = (1 - shrink) * Cj + shrink * C_all
        keep = pis > 1e-3
        if not keep.all():
            pis, mus, Cs = pis[keep] / pis[keep].sum(), mus[keep], [c for c, kk in zip(Cs, keep) if kk]
            k = len(pis)
    return [(float(p), m, c) for p, m, c in zip(pis, mus, Cs, strict=True)]


def laplace(face: Face, starts_u: list[np.ndarray]) -> dict:
    """Mode of the target in the face coordinates, Hessian, Laplace log evidence.

    (With a flat Dirichlet the ALR and logit-stick targets are the same function
    L(p) prod_i p_i, so this Laplace value does not depend on that choice.)"""
    u_hat, lp_hat = face.mode(starts_u)
    H = face.hessian(u_hat)
    ev = np.linalg.eigvalsh((H + H.T) / 2)
    Hpd = _pd(H, 1e-10)
    _s, logdet = np.linalg.slogdet(Hpd)
    return {"u_hat": u_hat, "lp_hat": lp_hat, "H": H, "H_min_eig": float(ev.min()),
            "cov": np.linalg.inv(Hpd),
            "logZ_laplace": float(lp_hat + face.dim / 2 * np.log(2 * np.pi) - 0.5 * logdet)}


def smc_draws(face: Face, lap: dict, rng: np.random.Generator, extra=(), chains: int = 100,
              steps: int = 8, target_ess: float = 200, max_evals: int = 20000) -> dict:
    """Posterior draws of a face by waste-free tempered SMC (:mod:`bio_sbi.inference.smc`)
    from a t reference (Laplace t, covariance x 1.44, plus ``extra`` t components)."""
    from bio_sbi.inference.nuts import ess
    from bio_sbi.inference.smc import tempered_smc

    comps = [(1.0, lap["u_hat"], lap["cov"] * 1.44)]
    if extra:
        comps = [(0.7, lap["u_hat"], lap["cov"] * 1.44)] + [
            (0.3 * c[0] / sum(e[0] for e in extra), c[1], c[2]) for c in extra]
    q = TMix(comps)
    res = tempered_smc(lambda U: face.targets(U), lambda r, k: q.sample(k, r), q.logpdf, rng,
                       chains=chains, steps=steps, max_evals=max_evals, target_ess=target_ess,
                       max_rhat=1.05, imh_components=3)  # fmt: skip
    info = res.info
    return {"u": res.draws, "lp": res.log_post.reshape(-1), "ess": float(ess(res.log_post)),
            "logZ_smc": float(info["log_evidence"]), "smc_status": info["status"],
            "smc_stages": int(info["stages"]), "smc_evals": int(info["evals"]),
            "smc_rhat": float(info.get("max_rhat", np.nan))}  # fmt: skip


def bridge_evidence(face: Face, U: np.ndarray, LP: np.ndarray, n1_eff: float,
                    rng: np.random.Generator, ks=(1, 4), n_fresh: int = 2000) -> dict:
    """Bridge sampling (Meng & Wong) from posterior draws ``U`` (log targets ``LP``), one
    estimate per proposal: a ``k``-component Gaussian mixture fitted to the draws, as t's
    (df 5, covariance x 1.21). Also the plain IS estimate from the same fresh draws."""
    ok = np.isfinite(LP)
    U, LP = U[ok], LP[ok]
    out = {}
    for k in ks:
        gm = weighted_gmm(U, np.ones(len(U)), k, rng)
        q = TMix([(c[0], c[1], c[2] * 1.21) for c in gm])
        xf = q.sample(n_fresh, rng)
        lwf = face.targets(xf) - q.logpdf(xf)
        lwf[~np.isfinite(lwf)] = -np.inf
        out[f"logZ_bridge_k{k}"] = bridge(LP - q.logpdf(U), lwf, n1_eff=max(n1_eff, 20.0))
        out[f"logZ_is_k{k}"] = float(logsumexp(lwf) - np.log(n_fresh))
        out[f"ess_is_k{k}"] = ess_is(lwf)
        out[f"pareto_k_k{k}"] = pareto_k(lwf)
    br = [out[f"logZ_bridge_k{k}"] for k in ks]
    out["logZ_bridge"] = float(np.mean(br))
    out["logZ_bridge_spread"] = float(np.max(br) - np.min(br))
    return out


def t_from_draws(u: np.ndarray, weight: float = 1.0, inflate: float = 1.21):
    return (weight, u.mean(0), np.cov(u.T) * inflate + 1e-6 * np.eye(u.shape[1]))


# ---------------------------------------------------------------------------- truncated Laplace
def laplace_physical(face: Face, mle_phys: np.ndarray, h: float = 1e-5, n_mc: int = 400_000,
                     rng: np.random.Generator | None = None) -> dict:
    """Laplace at the face MLE in physical coordinates (per row: all free entries but the
    largest), plain and boundary-aware.

    ``ll(t + delta) ~ ll(t) + g'delta - delta'H delta / 2`` (``g`` the gradient, nonzero at
    a boundary MLE; ``H`` from finite differences of the exact gradient, one-sided inward
    for entries at 0). The truncated version integrates this Gaussian over the simplex
    only: ``exp(ll + g'H^-1 g / 2) (2 pi)^(d/2) |H|^(-1/2) P(N(t + H^-1 g, H^-1) in simplex)``,
    the probability by Monte Carlo. The flat prior's density ((m-1)! per row) multiplies.
    """
    rng = rng or np.random.default_rng(0)
    lik = face.lik
    coords = []  # (block index, name index of coordinate, name index of reference)
    for bi, (_k, _free, idx, _b) in enumerate(face.blocks):
        ref = idx[int(np.argmax(mle_phys[idx]))]
        coords += [(bi, i, ref) for i in idx if i != ref]
    d = len(coords)
    O = lik.n_outcomes_max
    pos = {}  # name index -> (type row, entry)
    for k, t in enumerate(face.law.types):
        sl = face.space.row_slice(t)
        for e in range(sl.stop - sl.start):
            pos[sl.start + e] = (k, e)

    def P_of(phys):
        P = np.zeros((len(face.law.types), O))
        for i, (k, e) in pos.items():
            P[k, e] = phys[i]
        return P

    dP = np.zeros((d, O))
    rows = np.zeros(d, dtype=np.int64)
    for j, (_bi, i, ref) in enumerate(coords):
        k, e = pos[i]
        dP[j, e] += 1.0
        dP[j, pos[ref][1]] -= 1.0
        rows[j] = k

    def grad(phys):
        return lik.log_likelihood_P(P_of(phys), dP, rows, grad=True)

    t = np.asarray(mle_phys, dtype=float).copy()
    ll0, g0 = grad(t)
    H = np.zeros((d, d))
    for j, (_bi, i, ref) in enumerate(coords):
        e = np.zeros_like(t)
        e[i], e[ref] = h, -h
        if t[i] < h:  # at the boundary: one-sided, inward
            H[:, j] = (grad(t + e)[1] - g0) / h
        else:
            H[:, j] = (grad(t + e)[1] - grad(t - e)[1]) / (2 * h)
    H = -(H + H.T) / 2
    ev = np.linalg.eigvalsh(H)
    Hpd = _pd(H, 1e-10)
    _s, logdet = np.linalg.slogdet(Hpd)
    C = np.linalg.inv(Hpd)
    shift = C @ g0
    plain = ll0 + face.log_prior_phys + d / 2 * np.log(2 * np.pi) - 0.5 * logdet
    # probability that N(t + shift, C) lies in the simplex (free coords >= 0, row sums <= 1)
    x0 = np.array([t[i] for _bi, i, _r in coords]) + shift
    Z = x0 + rng.standard_normal((n_mc, d)) @ np.linalg.cholesky(C).T
    ok = np.all(Z >= 0, axis=1)
    for bi in range(len(face.blocks)):
        cols = [j for j, c in enumerate(coords) if c[0] == bi]
        ok &= Z[:, cols].sum(1) <= 1
    prob = ok.mean()
    trunc = plain + 0.5 * g0 @ shift + np.log(max(prob, 1.0 / n_mc))
    # simpler variant: Gaussian centred at the MLE (no gradient term) times its mass inside
    Z0 = Z - shift
    ok0 = np.all(Z0 >= 0, axis=1)
    for bi in range(len(face.blocks)):
        cols = [j for j, c in enumerate(coords) if c[0] == bi]
        ok0 &= Z0[:, cols].sum(1) <= 1
    trunc0 = plain + np.log(max(ok0.mean(), 1.0 / n_mc))
    at0 = [face.names[i] for _bi, i, _r in coords if t[i] < 1e-6]
    return {"logZ_laplace_phys": float(plain), "logZ_laplace_trunc": float(trunc),
            "logZ_laplace_trunc0": float(trunc0), "trunc0_prob": float(ok0.mean()),
            "trunc_prob": float(prob), "grad_term": float(0.5 * g0 @ shift),
            "H_min_eig_phys": float(ev.min()), "at_zero": at0, "ll_mle": float(ll0)}  # fmt: skip
