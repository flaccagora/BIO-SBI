"""Expected Fisher information of the exact two-type GW clone law and the Jeffreys prior.

For one clone observed at generation ``h`` (counts ``(A, B)``, conditioned on detection for
``selection=detected``) the expected Fisher information in the inference coordinates
``theta`` (the stick-breaking coordinates of the parameter space, the sampler's
coordinates) is

    I_h(theta) = sum_{c != 0} p~_c s_c s_c^T,   p~_c = p_c / q,   s_c = dp_c / p_c - dq / q,

with ``p`` the unconditional law (:func:`~bio_sbi.inference.gw_exact.law_and_jac`, exact
forward-mode derivatives) and ``q = P(Z_h != 0)``
(:func:`~bio_sbi.inference.gw_exact.detection_and_grad`). For a design with ``n_h`` clones
per horizon the information of the data is ``sum_h n_h I_h``; with equal ``n_h`` the
Jeffreys prior ``p_J(theta) ∝ sqrt(det sum_h I_h(theta))`` does not depend on ``n``.
Jeffreys is invariant: the density in any other coordinates follows by the Jacobian, so
``p_J`` in ``theta`` is directly comparable to the stick-breaking Dirichlet (Beta) density
the samplers use.

Grid: the law must be summed over (almost) all its mass at ``theta``, not only over
observed counts. The total ``N = A + B`` of a clone is at most ``s_max * d^h`` (largest
founder size times largest number of daughters to the ``h``); its exact law is a cheap 1-D
FFT (:func:`total_count_law`). The square grid ``L x L`` is the smallest FFT-friendly size
with ``P(N >= L | detected) <= tail_tol`` (capped at the full support), so the mass the
grid misses - and any aliased mass - is at most ``tail_tol``; no damping (``r = 1``), so
probabilities carry absolute round-off ~1e-16. Cells with ``p~ <= threshold`` or
unconditional ``p <= floor`` (1e-13, well above round-off) are left out of the sum (their
mass, reported as ``omitted_mass``, is negligible and their ratio ``dp / p`` is round-off
dominated). Where detection is below ``min_detection`` (1e-6, the floor of the exact
likelihood) the conditional law is round-off dominated: the information is undefined (NaN,
``log p_J = -inf``), i.e. the prior is restricted to ``P(detected) >= 1e-6`` at every horizon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.fft

from .gw_exact import GridSpec, GWModel, detection_and_grad, law_and_jac


def support_bound(model: GWModel, generations: int) -> int:
    """Largest possible clone total ``N`` after ``generations`` (plus one)."""
    d = int(np.max(np.asarray(model.offspring).sum(axis=2)))
    s = int(np.max(np.asarray(model.sizes)[np.asarray(model.size_probs) > 0]))
    return s * max(d, 1) ** int(generations) + 1


def total_count_law(model: GWModel, P: np.ndarray, generations: int) -> np.ndarray:
    """Exact unconditional law of ``N = A + B`` (labelled cells) on ``0..support_bound-1``.

    The pgf of ``N`` is ``G(z, z)``: a 1-D DFT of length ``support_bound`` has no aliasing.
    """
    L = scipy.fft.next_fast_len(support_bound(model, generations))
    z = np.exp(-2j * np.pi * np.arange(L // 2 + 1) / L)
    c0, cx, cy = model.f0
    F = [c0[k] + (cx[k] + cy[k]) * z for k in range(2)]
    off = np.asarray(model.offspring)
    for _ in range(int(generations)):
        F = [
            sum(
                P[k, o] * F[0] ** off[k, o, 0] * F[1] ** off[k, o, 1]
                for o in range(int(model.n_out[k]))
                if P[k, o] != 0
            )
            + 0j * z
            for k in range(2)
        ]
    S = model.founder[0] * F[0] + model.founder[1] * F[1]
    G = sum(pi * S**m for m, pi in zip(model.sizes, model.size_probs, strict=True) if pi)
    return scipy.fft.irfft(G, n=L)


def grid_size(model: GWModel, P: np.ndarray, generations: int, tail_tol: float = 1e-12,
              min_size: int = 16) -> tuple[int, float]:  # fmt: skip
    """Square grid side ``L`` with ``P(N >= L | detected) <= tail_tol`` and that tail mass.

    Returns ``(L, tail)``; ``L`` is 5-smooth (fast FFT) and at most the full support (then
    ``tail = 0``: the grid holds the whole law).
    """
    pN = np.clip(total_count_law(model, P, generations), 0.0, None)
    full = support_bound(model, generations)
    norm = pN[1:].sum() if model.detected else pN.sum()
    tails = np.cumsum(pN[::-1])[::-1] / max(norm, 1e-300)  # tails[n] = P(N >= n)
    need = int(np.argmax(tails <= tail_tol)) if np.any(tails <= tail_tol) else len(pN)
    need = max(need, int(min_size))
    if need >= full:
        return scipy.fft.next_fast_len(full), 0.0
    L = scipy.fft.next_fast_len(need)
    if L >= full:
        return scipy.fft.next_fast_len(full), 0.0
    return L, float(tails[L]) if L < len(tails) else 0.0


@dataclass
class HorizonInformation:
    """Fisher information of one clone at one horizon, with grid diagnostics."""

    generations: int
    info: np.ndarray
    detection: float
    grid: int
    tail_mass: float
    omitted_mass: float
    cells: int


def horizon_information(model: GWModel, P: np.ndarray, dP: np.ndarray, rows: np.ndarray,
                        generations: int, tail_tol: float = 1e-12, threshold: float = 1e-14,
                        grid: int | None = None, floor: float = 1e-13,
                        min_detection: float = 1e-6) -> HorizonInformation:  # fmt: skip
    """Return the expected Fisher information ``(D, D)`` of one clone at ``generations``.

    ``dP``/``rows`` as for :func:`~bio_sbi.inference.gw_exact.law_and_jac` (one direction
    per coordinate). Conditioned on detection if ``model.detected``. ``grid`` overrides the
    automatic grid side (:func:`grid_size`; validation).
    """
    if grid is None:
        L, tail = grid_size(model, P, generations, tail_tol)
    else:
        L, tail = int(grid), float("nan")
    p, dp = law_and_jac(model, P, dP, rows, generations, GridSpec(L, L, 1.0, 1.0))
    D = dP.shape[0]
    if model.detected:
        q, dq = detection_and_grad(model, P, dP, rows, generations)
        p[0, 0] = 0.0
    else:
        q, dq = 1.0, np.zeros(D)
    pc = p / q
    keep = (pc > threshold) & (p > floor)
    w = pc[keep]
    s = dp[:, keep] / p[keep][None, :] - (dq / q)[:, None]
    info = (s * w[None, :]) @ s.T
    info = 0.5 * (info + info.T)
    if q < min_detection:
        info = np.full_like(info, np.nan)
    return HorizonInformation(int(generations), info, float(q), L, tail,
                              float(max(1.0 - w.sum(), 0.0)), int(keep.sum()))  # fmt: skip


@dataclass
class Information:
    """Fisher information summed over horizons (``weights`` per horizon) at one ``theta``."""

    info: np.ndarray
    horizons: list[HorizonInformation] = field(default_factory=list)

    @property
    def logdet(self) -> float:
        """``log det info`` (``-inf`` if singular or not positive definite)."""
        if not np.all(np.isfinite(self.info)):
            return -math.inf
        sign, ld = np.linalg.slogdet(self.info)
        return float(ld) if sign > 0 else -math.inf

    @property
    def min_eigenvalue(self) -> float:
        """Smallest eigenvalue of the (symmetrised) information."""
        if not np.all(np.isfinite(self.info)):
            return math.nan
        return float(np.linalg.eigvalsh(self.info)[0])


class JeffreysPrior:
    """Jeffreys prior of the exact GW model for a multi-horizon design, in ``theta``.

    ``p_J(theta) ∝ sqrt(det sum_h w_h I_h(theta))`` with ``I_h`` the information of one
    clone at horizon ``h`` (:func:`horizon_information`) and ``w_h`` the relative number of
    clones per horizon (default equal). ``theta`` = the sampler's stick coordinates of
    ``simulator.space`` (all free rows; founder rows fixed). Unnormalised.

    Args:
        simulator: Galton-Watson simulator (inference model).
        observation: Observation model (capture, confusion, selection).
        horizons: Measurement endpoints of the design.
        weights: Relative clone numbers per horizon (default 1 each).
        tail_tol: Mass of ``N`` allowed beyond the grid (:func:`grid_size`).
        threshold: Conditional cell probability below which cells are left out.
        floor: Unconditional cell probability below which cells are left out.
        min_detection: Smallest detection probability (any horizon) with a defined prior.
    """

    def __init__(self, simulator, observation, horizons, weights=None, tail_tol: float = 1e-12,
                 threshold: float = 1e-14, floor: float = 1e-13,
                 min_detection: float = 1e-6) -> None:  # fmt: skip
        from ..summaries import Cohort
        from .exact_gw import ExactLikelihood

        # a one-clone dataset per horizon: only the model, the coordinates and the Jacobian
        # of the likelihood object are used
        dummy = [Cohort(float(h), np.array([[1, 0]])) for h in horizons]
        self.lik = ExactLikelihood(simulator, observation, dummy)
        self.model = self.lik.model
        self.generations = [h.generations for h in self.lik.data]
        self.weights = np.ones(len(horizons)) if weights is None else np.asarray(weights, float)
        self.tail_tol, self.threshold = float(tail_tol), float(threshold)
        self.floor, self.min_detection = float(floor), float(min_detection)
        self.dim = self.lik.dim

    def information(self, theta: np.ndarray) -> Information:
        """Return the summed Fisher information at ``theta`` (with per-horizon diagnostics)."""
        P, dP, rows = self.lik.jacobian(np.asarray(theta, dtype=float))
        parts = [horizon_information(self.model, P, dP, rows, g, self.tail_tol, self.threshold,
                                     floor=self.floor, min_detection=self.min_detection)
                 for g in self.generations]  # fmt: skip
        info = sum(w * h.info for w, h in zip(self.weights, parts, strict=True))
        return Information(np.asarray(info), parts)

    def log_density(self, theta: np.ndarray) -> float:
        """Unnormalised ``log p_J(theta) = 0.5 log det I(theta)``."""
        return 0.5 * self.information(theta).logdet

    def log_dirichlet(self, theta: np.ndarray, alpha: float | np.ndarray = 0.5) -> float:
        """Log density in ``theta`` of a Dirichlet prior on every free row (normalised).

        ``alpha``: concentration per free entry (scalar = symmetric). The stick of entry
        ``j`` of a row is ``Beta(alpha_j, sum_{l > j} alpha_l)``.
        """
        from scipy.special import betaln

        theta = np.asarray(theta, dtype=float)
        a, b = self.stick_beta(alpha)
        return float(np.sum((a - 1) * np.log(theta) + (b - 1) * np.log1p(-theta) - betaln(a, b)))

    def stick_beta(self, alpha: float | np.ndarray = 0.5) -> tuple[np.ndarray, np.ndarray]:
        """Beta parameters ``(a, b)`` per stick of a Dirichlet(``alpha``) on every free row."""
        a, b = [], []
        for r in self.lik.space.rows:
            if not r.n_sticks:
                continue
            al = np.broadcast_to(np.asarray(alpha, dtype=float), (len(r.free),))
            tail = np.cumsum(al[::-1])[::-1]
            a.extend(al[:-1])
            b.extend(tail[1:])
        return np.asarray(a), np.asarray(b)


# ---------------------------------------------------------------------- batched (torch)
def _torch_law_and_jac(model: GWModel, C, dC, rows, mono, generations: int, L: int, device):
    """Batched unconditional law ``p (B, L, L)`` and ``dp (B, D, L, L)`` (torch, r = 1).

    Polynomial form of the offspring pgfs (``C (B, 2, M)``, direction rows ``dC (B, D, M)``;
    :func:`~bio_sbi.inference.gw_exact.poly_coefficients`), forward-mode derivatives - the
    same recursion as :func:`~bio_sbi.inference.gw_exact.law_and_jac`, vectorised over the
    batch and the grid.
    """
    import torch

    ma, mb, _, maxp = mono
    ma, mb = [int(x) for x in ma], [int(x) for x in mb]
    dt = torch.complex128
    C = torch.as_tensor(C, device=device, dtype=torch.float64)
    dC = torch.as_tensor(dC, device=device, dtype=torch.float64)
    rows = torch.as_tensor(np.asarray(rows), device=device)
    rA = (rows == 0).to(torch.float64)[None, :, None, None]
    rB = (rows == 1).to(torch.float64)[None, :, None, None]
    Bn, D = dC.shape[0], dC.shape[1]
    k = torch.arange(L, device=device, dtype=torch.float64)
    zx = torch.exp(-2j * math.pi * k / L).to(dt)[:, None]
    zy = torch.exp(-2j * math.pi * k[: L // 2 + 1] / L).to(dt)[None, :]
    c0, cx, cy = model.f0
    FA = (c0[0] + cx[0] * zx + cy[0] * zy).expand(Bn, L, L // 2 + 1).clone()
    FB = (c0[1] + cx[1] * zx + cy[1] * zy).expand(Bn, L, L // 2 + 1).clone()
    dA = torch.zeros((Bn, D) + FA.shape[1:], dtype=dt, device=device)
    dB = torch.zeros_like(dA)

    def coef(x):  # (B,) real -> broadcastable over the grid
        return x[:, None, None]

    for _ in range(int(generations)):
        pa = [torch.ones_like(FA)]
        pb = [torch.ones_like(FB)]
        for _e in range(maxp):
            pa.append(pa[-1] * FA)
            pb.append(pb[-1] * FB)
        mon = torch.stack([pa[a] * pb[b] for a, b in zip(ma, mb, strict=True)], 1)  # B,M,X,Y
        fA = torch.einsum("bm,bmxy->bxy", C[:, 0].to(dt), mon)
        fB = torch.einsum("bm,bmxy->bxy", C[:, 1].to(dt), mon)
        jac = [[0, 0], [0, 0]]  # jac[k][l] = d f_k / d F_l
        for kk in range(2):
            gx = sum(coef(C[:, kk, m]) * a * pa[a - 1] * pb[b]
                     for m, (a, b) in enumerate(zip(ma, mb, strict=True)) if a > 0)  # fmt: skip
            gy = sum(coef(C[:, kk, m]) * b * pa[a] * pb[b - 1]
                     for m, (a, b) in enumerate(zip(ma, mb, strict=True)) if b > 0)  # fmt: skip
            jac[kk] = [gx, gy]
        t = torch.einsum("bdm,bmxy->bdxy", dC.to(dt), mon)
        nA = jac[0][0][:, None] * dA + jac[0][1][:, None] * dB + rA * t
        nB = jac[1][0][:, None] * dA + jac[1][1][:, None] * dB + rB * t
        dA, dB, FA, FB = nA, nB, fA, fB
        del mon, t, pa, pb, jac
    S = model.founder[0] * FA + model.founder[1] * FB
    dS = model.founder[0] * dA + model.founder[1] * dB
    G = torch.zeros_like(S)
    dG = torch.zeros_like(S)
    for m, pi in zip(model.sizes, model.size_probs, strict=True):
        if pi:
            G = G + float(pi) * S ** int(m)
            dG = dG + float(pi) * int(m) * S ** (int(m) - 1)
    p = torch.fft.irfft2(G, s=(L, L))
    dp = torch.fft.irfft2(dG[:, None] * dS, s=(L, L))
    return p, dp


_CUDA_KERNEL = None
_MAXD = 32  # max directions (coordinates) of the fused kernel
_MAXM = 8  # max distinct daughter monomials


def _cuda_kernel():
    """Fused numba.cuda forward-mode kernel (compiled on first use)."""
    global _CUDA_KERNEL
    if _CUDA_KERNEL is not None:
        return _CUDA_KERNEL
    from numba import complex128, cuda

    @cuda.jit(fastmath=False)
    def kernel(C, dC, rows, ma, mb, gens, zx, zy, F0c, F0x, F0y, phi, sizes, pis, out, dout):
        t = cuda.grid(1)
        b = cuda.blockIdx.y
        n = zy.shape[0]
        if t >= zx.shape[0] * n:
            return
        i = t // n
        j = t - i * n
        x = zx[i]
        y = zy[j]
        FA = F0c[0] + F0x[0] * x + F0y[0] * y
        FB = F0c[1] + F0x[1] * x + F0y[1] * y
        D = dC.shape[1]
        M = ma.shape[0]
        dA = cuda.local.array(_MAXD, complex128)
        dB = cuda.local.array(_MAXD, complex128)
        mono = cuda.local.array(_MAXM, complex128)
        for d in range(D):
            dA[d] = 0j
            dB[d] = 0j
        one = 1.0 + 0j
        for _ in range(gens):
            A2 = FA * FA
            B2 = FB * FB
            fA = 0j
            fB = 0j
            jAA = 0j
            jAB = 0j
            jBA = 0j
            jBB = 0j
            for m in range(M):
                a = ma[m]
                bb = mb[m]
                pa = one if a == 0 else (FA if a == 1 else A2)
                pb = one if bb == 0 else (FB if bb == 1 else B2)
                mo = pa * pb
                mono[m] = mo
                c0 = C[b, 0, m]
                c1 = C[b, 1, m]
                fA += c0 * mo
                fB += c1 * mo
                if a > 0:
                    ga = a * (one if a == 1 else FA) * pb
                    jAA += c0 * ga
                    jBA += c1 * ga
                if bb > 0:
                    gb = bb * pa * (one if bb == 1 else FB)
                    jAB += c0 * gb
                    jBB += c1 * gb
            for d in range(D):
                tt = 0j
                for m in range(M):
                    tt += dC[b, d, m] * mono[m]
                nA = jAA * dA[d] + jAB * dB[d]
                nB = jBA * dA[d] + jBB * dB[d]
                if rows[d] == 0:
                    nA += tt
                else:
                    nB += tt
                dA[d] = nA
                dB[d] = nB
            FA = fA
            FB = fB
        S = phi[0] * FA + phi[1] * FB
        G = 0j
        dG = 0j
        for s in range(sizes.shape[0]):
            k = sizes[s]
            G += pis[s] * S**k
            if k > 0:
                dG += pis[s] * k * S ** (k - 1)
        out[b, i, j] = G
        for d in range(D):
            dout[b, d, i, j] = dG * (phi[0] * dA[d] + phi[1] * dB[d])

    _CUDA_KERNEL = kernel
    return kernel


def _cuda_law_and_jac(model: GWModel, C, dC, rows, mono, generations: int, L: int, device):
    """:func:`_torch_law_and_jac` with the pgf recursion in one fused CUDA kernel."""
    import torch
    from numba import cuda

    ma, mb, _, maxp = mono
    if maxp > 2 or dC.shape[1] > _MAXD or len(ma) > _MAXM:
        raise ValueError("fused kernel: at most binary divisions, 32 coordinates, 8 monomials")
    kernel = _cuda_kernel()
    dev = torch.device(device)
    Bn, D = dC.shape[0], dC.shape[1]
    Ly2 = L // 2 + 1
    k = np.arange(L)
    zx = np.exp(-2j * np.pi * k / L)
    zy = np.exp(-2j * np.pi * k[:Ly2] / L)
    out = torch.empty((Bn, L, Ly2), dtype=torch.complex128, device=dev)
    dout = torch.empty((Bn, D, L, Ly2), dtype=torch.complex128, device=dev)
    c0, cx, cy = model.f0

    def arr(x, dtype):
        return cuda.as_cuda_array(torch.as_tensor(np.ascontiguousarray(x, dtype=dtype), device=dev))

    tpb = 128
    grid = ((L * Ly2 + tpb - 1) // tpb, Bn)
    kernel[grid, tpb](
        arr(C, np.float64), arr(dC, np.float64), arr(rows, np.int64), arr(ma, np.int64),
        arr(mb, np.int64), int(generations), arr(zx, np.complex128), arr(zy, np.complex128),
        arr(c0, np.float64), arr(cx, np.float64), arr(cy, np.float64),
        arr(model.founder, np.float64), arr(model.sizes, np.int64),
        arr(model.size_probs, np.float64), cuda.as_cuda_array(out), cuda.as_cuda_array(dout),
    )  # fmt: skip
    p = torch.fft.irfft2(out, s=(L, L))
    dp = torch.fft.irfft2(dout, s=(L, L))
    return p, dp


def batch_information(prior: JeffreysPrior, thetas: np.ndarray, device: str = "cuda",
                      max_batch: int = 8, max_elements: float = 4e7,
                      backend: str | None = None) -> list[Information]:  # fmt: skip
    """:meth:`JeffreysPrior.information` for many ``thetas`` at once on a torch ``device``.

    Same quantities (grid per horizon = the largest :func:`grid_size` of the batch, so
    at least as accurate); vectors are batched by grid size, at most ``max_batch`` and
    ``max_elements`` derivative-grid elements (``B D L^2``) per call. ``backend``:
    ``"cuda"`` (fused numba.cuda kernel; default on a CUDA device) or ``"torch"``
    (elementwise torch ops; any device).
    """
    import torch

    from .gw_exact import monomials, poly_coefficients

    backend = backend or ("cuda" if str(device).startswith("cuda") else "torch")
    law = _cuda_law_and_jac if backend == "cuda" else _torch_law_and_jac
    lik, model = prior.lik, prior.model
    mono = monomials(model)
    thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
    jacs = [lik.jacobian(t) for t in thetas]
    out_parts: list[list[HorizonInformation | None]] = [[None] * len(prior.generations)
                                                        for _ in thetas]  # fmt: skip
    for hi, g in enumerate(prior.generations):
        sizes = [grid_size(model, P, g, prior.tail_tol) for P, _, _ in jacs]
        order = np.argsort([s[0] for s in sizes], kind="stable")
        i = 0
        while i < len(order):
            # ascending grid sizes: the batch's grid is that of its last member
            e = i + 1
            while (e < len(order) and e - i < max_batch
                   and (e - i + 1) * prior.dim * sizes[order[e]][0] ** 2 <= max_elements):  # fmt: skip
                e += 1
            idx = list(order[i:e])
            L = max(sizes[j][0] for j in idx)
            i = e
            CdC = [poly_coefficients(mono[2], len(mono[0]), *jacs[j]) for j in idx]
            rows = jacs[idx[0]][2]
            C = np.stack([c for c, _ in CdC])
            dC = np.stack([d for _, d in CdC])
            det = [detection_and_grad(model, *jacs[j], g) if model.detected
                   else (1.0, np.zeros(prior.dim)) for j in idx]  # fmt: skip
            with torch.no_grad():
                p, dp = law(model, C, dC, rows, mono, g, L, device)
                p_dev = p.device
                q = torch.as_tensor([d[0] for d in det], device=p.device)[:, None, None]
                with np.errstate(divide="ignore", invalid="ignore"):  # q = 0: info NaN
                    dqq = np.stack([d[1] / d[0] for d in det])
                dqq = torch.as_tensor(dqq, device=p.device)
                if model.detected:
                    p[:, 0, 0] = 0.0
                pc = p / q
                keep = (pc > prior.threshold) & (p > prior.floor)
                w = torch.where(keep, pc, 0.0)
                inv = torch.where(keep, 1.0 / torch.where(keep, p, 1.0), 0.0)
                s = (dp * inv[:, None] - dqq[:, :, None, None]).flatten(2)
                s.mul_(w.flatten(1).sqrt()[:, None])  # in place: s sqrt(w)
                info = torch.bmm(s, s.transpose(1, 2)).cpu().numpy()
                for bi in range(len(idx)):
                    if model.detected and det[bi][0] < prior.min_detection:
                        info[bi] = np.nan
                mass = w.flatten(1).sum(1).cpu().numpy()
                cells = keep.flatten(1).sum(1).cpu().numpy()
                del p, dp, s, pc, keep, w, inv
                if p_dev.type == "cuda":
                    torch.cuda.empty_cache()  # share the device with other processes
            for bi, j in enumerate(idx):
                tail = sizes[j][1] if L == sizes[j][0] else float("nan")
                out_parts[j][hi] = HorizonInformation(
                    int(g), 0.5 * (info[bi] + info[bi].T), float(det[bi][0]), int(L), tail,
                    float(max(1.0 - mass[bi], 0.0)), int(cells[bi]),
                )  # fmt: skip
    res = []
    for parts in out_parts:
        info = sum(w * h.info for w, h in zip(prior.weights, parts, strict=True))
        res.append(Information(np.asarray(info), parts))
    return res
