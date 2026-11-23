"""Sampler coordinates for ``exact_gw``: stick order per row and logit / probit links.

``exact_gw`` samples ``u = logit(theta)``, ``theta`` the stick-breaking coordinates of
:class:`~bio_sbi.simulators.law.ParameterSpace` (each free row broken in its outcome-table
order, sticks ``Beta(alpha_i, sum_{j>i} alpha_j)``: the Dirichlet prior). Two weak
directions of the two-type posterior are curved in these coordinates (B's row: "B switches
back to A" against "B is lost", the latter spread over death and three divisions), and
probabilities the data allow to be ~0 have exponential tails towards ``u -> -inf``.
:class:`StickCoordinates` is a bijection ``v -> theta`` with exact log Jacobian; NUTS
targets ``log p(theta(v)) + log |d theta / d v|``, so the posterior of ``theta`` (and of the
physical probabilities) is unchanged. ``spec`` is one of :data:`SPECS`:

* ``brow`` - break B's row (the second type; generally ``types[1]``) in the order
  ``to_A, persist, death, div_AA, div_AB, div_BB``: stick 1 = ``P(B -> A)``, stick 2 =
  persist share of the rest, the remaining "B is lost" mass is split by stick 3 into death
  and the divisions, which sticks 4 and 5 split into AA / AB / BB (hierarchical sticks).
  For a Dirichlet row, stick breaking in *any* order with ``Beta(alpha_(i),
  sum_{j>i} alpha_(j))`` sticks is again exactly that Dirichlet (aggregation property), so
  the prior is unchanged for every concentration vector (e.g. ``flat_btoa``: to_A 1,
  others 0.2). Generally ``stick_order = {row: [entries...]}`` (listed free entries first,
  in that order, then the unlisted free ones in table order).
* ``probit`` - ``stick = Phi(v)`` instead of ``expit(v)``: ``log |d stick / d v| = log
  phi(v)`` (Gaussian instead of exponential tails at the simplex boundary).
* ``brow+probit`` - both.

The likelihood gradient is taken directly with respect to the new sticks (the kernel
propagates ``dP / d stick``), so a non-default order costs nothing extra. ``none`` keeps
the original code path (``ExactLikelihood.log_posterior_u``), bit for bit.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import expit, log_expit, log_ndtr, logit, ndtr, ndtri, ndtri_exp

#: Recognised specs.
SPECS = ("none", "probit", "brow", "brow+probit")
#: |v| bound of the initial optimisation per link (stick within ~1e-7 / 6e-16 of 0 or 1).
BOUND = {"logit": 15.0, "probit": 8.0}
_LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


def parse_spec(spec: str | None) -> tuple[str, bool]:
    """``(link, brow)`` of a spec; raises on unknown specs."""
    spec = str(spec or "none")
    parts = set(spec.replace(" ", "").split("+")) - {"none", ""}
    unknown = parts - {"probit", "brow", "logit"}
    if unknown or ("probit" in parts and "logit" in parts):
        raise ValueError(f"exact_gw: unknown reparam {spec!r} ({' | '.join(SPECS)})")
    return ("probit" if "probit" in parts else "logit"), "brow" in parts


def brow_order(types) -> dict[str, list[str]]:
    """Stick order of the second type's row: switch back, persist, death, divisions."""
    a, b = types[0], types[1]
    return {b: [f"to_{a}", "persist", "death", f"div_{a}{a}", f"div_{a}{b}", f"div_{b}{b}"]}


def _break(p: np.ndarray) -> np.ndarray:
    """Sticks ``p_i / sum_{j>=i} p_j`` of normalised rows ``p (..., m)`` (0.5 if undefined)."""
    remaining = np.cumsum(p[..., ::-1], axis=-1)[..., ::-1][..., :-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(remaining > 0, p[..., :-1] / np.maximum(remaining, 1e-300), 0.5)


def _beta_params(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tail = np.cumsum(alpha[::-1])[::-1]
    return alpha[:-1].astype(float), tail[1:].astype(float)


class StickCoordinates:
    """Bijection ``v -> theta`` (sampler coordinates -> stick coordinates of the space).

    Args:
        lik: :class:`~bio_sbi.inference.exact_gw.ExactLikelihood`.
        spec: One of :data:`SPECS` (``+``-combinations of ``probit`` and ``brow``).
        stick_order: ``{row prefix: [entry, ...]}`` order overrides (applied after
            ``brow``).
    """

    def __init__(self, lik, spec: str | None = "none", stick_order: dict | None = None) -> None:
        self.lik = lik
        self.space = space = lik.space
        self.spec = str(spec or "none")
        self.link, brow = parse_spec(spec)
        if self.link == "logit" and not brow and not stick_order:
            # default coordinates: delegate to lik.log_posterior_u (the original code path)
            self.identity, self.bound, self.dim = True, BOUND["logit"], lik.dim
            self.a, self.b = np.asarray(lik.prior_a, float), np.asarray(lik.prior_b, float)
            self.orders: dict[str, list[str]] = {}
            return
        if getattr(lik, "founder_free", False):
            raise ValueError("exact_gw reparam / stick_order: free founder rows need the "
                             "default coordinates (reparam: none)")  # fmt: skip
        types = list(lik.sim.law.types)
        orders: dict = brow_order(types) if brow else {}
        orders.update({str(k): list(v) for k, v in (stick_order or {}).items()})
        unknown = set(orders) - {r.prefix for r in space.rows if r.n_sticks}
        if unknown:
            raise ValueError(f"exact_gw stick_order: no free row {sorted(unknown)}")
        # per free row: (prefix, type index, outcome columns in stick order, mass, v slice)
        self.blocks: list[tuple[str, int, np.ndarray, float, slice]] = []
        self.orders = {}
        a_all, b_all, start, identity_order = [], [], 0, True
        self._P_fixed = np.zeros((len(types), lik.n_outcomes_max))
        self._fixed_phys = np.zeros(len(space.names))
        for r in space.rows:
            sl = space.row_slice(r.prefix)
            # fixed entries, and a lone free entry (no stick) taking the leftover mass
            known = dict(r.fixed)
            if len(r.free) == 1:
                known[r.free[0]] = r.mass
            for i, val in known.items():
                self._fixed_phys[sl.start + i] = val
                if r.prefix in types:
                    self._P_fixed[types.index(r.prefix), i] = val
            if not r.n_sticks:
                continue
            free = list(r.free)
            want = orders.get(r.prefix)
            if want is not None:
                bad = set(want) - set(r.entries)
                if bad:
                    raise ValueError(f"exact_gw stick_order: row {r.prefix} has no {sorted(bad)}")
                listed = [r.entries.index(e) for e in want if r.entries.index(e) in free]
                order = listed + [i for i in free if i not in listed]
            else:
                order = free
            identity_order &= order == free
            perm = [free.index(i) for i in order]
            a, b = _beta_params(np.asarray(r.alpha, float)[perm])
            a_all.extend(a)
            b_all.extend(b)
            m = len(order)
            cols = sl.start + np.asarray(order, np.int64)  # physical-vector index
            self.blocks.append((r.prefix, types.index(r.prefix), np.asarray(order, np.int64),
                                float(r.mass), slice(start, start + m - 1), cols))  # fmt: skip
            self.orders[r.prefix] = [r.entries[i] for i in order]
            start += m - 1
        self.dim = start
        if self.dim != lik.dim:
            raise ValueError("exact_gw reparam: free scalar parameters are not supported")
        self.a, self.b = np.asarray(a_all), np.asarray(b_all)
        self.identity = self.link == "logit" and identity_order
        self.bound = BOUND[self.link]

    # ---------------------------------------------------------------- link
    def sticks(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(s, 1 - s)`` (the complement computed without cancellation)."""
        v = np.asarray(v, float)
        if self.link == "logit":
            return expit(v), expit(-v)
        return ndtr(v), ndtr(-v)

    def from_sticks(self, s: np.ndarray) -> np.ndarray:
        """``v`` of sticks ``s`` (inverse link)."""
        s = np.asarray(s, float)
        if self.link == "logit":
            return logit(s)
        return ndtri(s)

    def log_dsdv(self, v: np.ndarray) -> np.ndarray:
        """``log |d s / d v|`` per coordinate."""
        v = np.asarray(v, float)
        if self.link == "logit":
            return log_expit(v) + log_expit(-v)
        return -0.5 * v**2 - _LOG_SQRT_2PI

    # ---------------------------------------------------------------- maps
    def physical(self, v: np.ndarray) -> np.ndarray:
        """Physical parameters ``(..., n_names)`` at ``v (..., d)`` (fixed entries filled)."""
        v = np.asarray(v, float)
        if self.identity:
            return self.space.decode(expit(v))
        batch = v.shape[:-1]
        s, sc = self.sticks(v)
        out = np.broadcast_to(self._fixed_phys, (*batch, len(self._fixed_phys))).copy()
        for _, _, _, mass, vs, cols in self.blocks:
            ss, cc = s[..., vs], sc[..., vs]
            tails = np.concatenate([np.ones((*batch, 1)), np.cumprod(cc, axis=-1)], -1)
            out[..., cols] = np.concatenate([ss * tails[..., :-1], tails[..., -1:]], -1) * mass
        return out

    def theta(self, v: np.ndarray) -> np.ndarray:
        """Stick coordinates of the space (its own order) at ``v (..., d)``."""
        if self.identity:
            return expit(np.asarray(v, float))
        phys = self.physical(v)
        out = []
        for r in self.space.rows:
            if r.n_sticks:
                sl = self.space.row_slice(r.prefix)
                out.append(_break(phys[..., sl][..., list(r.free)] / r.mass))
        return np.concatenate(out, axis=-1)

    def from_theta(self, theta: np.ndarray, stick_clip: float = 1e-12,
                   v_clip: float | None = None) -> np.ndarray:  # fmt: skip
        """``v`` of space stick coordinates ``theta (..., d)``.

        The new-order sticks are clipped to ``[stick_clip, 1 - stick_clip]`` and ``v`` to
        ``+-v_clip`` (default: the link's bound). With the default coordinates this is
        ``clip(logit(clip(theta)))`` as in the warm start.
        """
        theta = np.asarray(theta, float)
        v_clip = self.bound if v_clip is None else float(v_clip)
        if self.identity:
            t = np.clip(theta, stick_clip, 1 - stick_clip)
            return np.clip(np.log(t) - np.log1p(-t), -v_clip, v_clip)
        phys = self.space.decode(theta)
        s = np.concatenate([_break(phys[..., cols] / mass)
                            for _, _, _, mass, _, cols in self.blocks], axis=-1)  # fmt: skip
        s = np.clip(s, stick_clip, 1 - stick_clip)
        return np.clip(self.from_sticks(s), -v_clip, v_clip)

    # ---------------------------------------------------------------- density
    def log_prior(self, v: np.ndarray, normalised: bool = False) -> np.ndarray:
        """Log prior density of ``v`` (sticks' Betas times the link Jacobian), batched."""
        v = np.asarray(v, float)
        if self.link == "logit":
            lp = (self.a * log_expit(v) + self.b * log_expit(-v)).sum(-1)
        else:
            lp = ((self.a - 1) * log_ndtr(v) + (self.b - 1) * log_ndtr(-v)
                  - 0.5 * v**2 - _LOG_SQRT_2PI).sum(-1)  # fmt: skip
        if normalised:
            from scipy.special import betaln

            lp = lp - betaln(self.a, self.b).sum()
        return lp

    def sample_prior(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """``n`` exact prior draws of ``v`` (Betas in the stick order, inverse link).

        Computed in log space (``s = X / (X + Y)``, ``X ~ Gamma(a)``, ``Y ~ Gamma(b)``,
        ``log Gamma(a) = log Gamma(a + 1) + log U / a``): with concentrations < 1 a
        noticeable share of the sticks lies below ``1e-300`` and must not be clipped.
        """
        shape = (int(n), self.dim)

        def log_gamma(k):
            return np.log(rng.gamma(k + 1.0, size=shape)) + np.log(rng.uniform(size=shape)) / k

        lx, ly = log_gamma(self.a), log_gamma(self.b)
        if self.link == "logit":
            return lx - ly
        log_s = lx - np.logaddexp(lx, ly)
        log_c = ly - np.logaddexp(lx, ly)
        return np.where(log_s < log_c, ndtri_exp(log_s), -ndtri_exp(log_c))

    def P_and_dP(self, s: np.ndarray, sc: np.ndarray, grad: bool = True):
        """Outcome matrix ``P (2, O)``, ``dP / d s (d, O)`` and the row of each stick."""
        P = self._P_fixed.copy()
        dP = np.zeros((self.dim if grad else 0, P.shape[1]))
        rows = np.zeros(self.dim if grad else 0, dtype=np.int64)
        for _, k, order, mass, vs, _ in self.blocks:
            ss, cc = s[vs], sc[vs]
            m = len(order)
            prefix = np.concatenate([[1.0], np.cumprod(cc)])  # prod_{l<i} (1 - s_l)
            P[k, order] = np.append(ss * prefix[:-1], prefix[-1]) * mass
            if not grad:
                continue
            # p_i = mass s_i prod_{l<i} (1 - s_l)  (s_{m-1} := 1)
            for j in range(m - 1):
                t = vs.start + j
                rows[t] = k
                dP[t, order[j]] = mass * prefix[j]
                for i in range(j + 1, m):
                    si = ss[i] if i < m - 1 else 1.0
                    dP[t, order[i]] = -mass * si * np.prod(np.delete(cc[:i], j))
        return P, dP, rows

    def log_posterior(self, v: np.ndarray) -> tuple[float, np.ndarray]:
        """Log posterior density of ``v`` (Jacobian included) and its gradient."""
        if self.identity:
            return self.lik.log_posterior_u(v)
        v = np.asarray(v, float)
        s, sc = self.sticks(v)
        if np.any(s <= 0) or np.any(sc <= 0):
            return -np.inf, np.zeros_like(v)
        P, dP, rows = self.P_and_dP(s, sc)
        ll, g = self.lik.log_likelihood_P(P, dP, rows)
        if not np.isfinite(ll):
            return -np.inf, np.zeros_like(v)
        lp = ll + float(self.log_prior(v))
        if self.link == "logit":
            grad = g * s * sc + self.a * sc - self.b * s
        else:
            log_phi = -0.5 * v**2 - _LOG_SQRT_2PI
            mills_lo = np.exp(log_phi - log_ndtr(v))  # phi / Phi(v)
            mills_hi = np.exp(log_phi - log_ndtr(-v))  # phi / Phi(-v)
            grad = g * np.exp(log_phi) + (self.a - 1) * mills_lo - (self.b - 1) * mills_hi - v
        return lp, grad

    def log_posterior_value(self, v: np.ndarray) -> float:
        """Log posterior density of ``v`` without gradient (cheaper likelihood)."""
        v = np.asarray(v, float)
        if self.identity:
            theta = expit(v)
            if np.any(theta <= 0) or np.any(theta >= 1):
                return -np.inf
            ll, _ = self.lik.log_likelihood(theta, grad=False)
        else:
            s, sc = self.sticks(v)
            if np.any(s <= 0) or np.any(sc <= 0):
                return -np.inf
            P, dP, rows = self.P_and_dP(s, sc, grad=False)
            ll, _ = self.lik.log_likelihood_P(P, dP, rows, grad=False)
        if not np.isfinite(ll):
            return -np.inf
        return float(ll + self.log_prior(v))

    def info(self) -> dict:
        """Return the settings for the diagnostics."""
        return {"spec": self.spec, "link": self.link, "stick_order": self.orders}
