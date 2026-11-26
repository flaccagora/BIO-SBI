"""GPU prototype: exact GW log-likelihood + gradient for a batch of parameter vectors.

Same algorithm as ``kernel=adjoint`` (polynomial-form pgf recursion, reverse-mode
gradient through the adjoint of the inverse FFT), written with batched torch tensors
(complex128) so B parameter vectors (e.g. B chains in lockstep) share every kernel launch
and every FFT. Parameter transforms (theta -> P, dP) and the detection term run on the
CPU (Numba, per vector) and are timed separately.

Not used by the pipeline: a feasibility prototype (see bench_gpu.py).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from bio_sbi.inference import gw_exact as G


class TorchGW:
    """Batched exact GW likelihood of one ``ExactLikelihood`` dataset on ``device``."""

    def __init__(self, lik, device: str = "cuda"):
        self.lik = lik
        self.dev = torch.device(device)
        m = lik.model
        self.mono = G.monomials(m)
        ma, mb, _, maxp = self.mono
        self.ma, self.mb, self.maxp = [int(x) for x in ma], [int(x) for x in mb], maxp
        c0, cx, cy = m.f0
        self.phi = [float(x) for x in m.founder]
        self.sizes = [int(x) for x in m.sizes]
        self.pis = [float(x) for x in m.size_probs]
        self.hz = []
        for h in lik.data:
            g = h.grid
            zx, zy = G._grid_points(g)
            zx = torch.as_tensor(zx, device=self.dev)[:, None]
            zy = torch.as_tensor(zy, device=self.dev)[None, :]
            F0A = c0[0] + cx[0] * zx + cy[0] * zy
            F0B = c0[1] + cx[1] * zx + cy[1] * zy
            n = zy.shape[1]
            wj = torch.full((n,), 2.0, dtype=torch.float64, device=self.dev)
            wj[0] = 1.0
            if g.Ly % 2 == 0:
                wj[-1] = 1.0
            amp = g.rx ** -h.a.astype(float) * g.ry ** -h.b.astype(float)
            self.hz.append({
                "gens": int(h.generations), "Lx": g.Lx, "Ly": g.Ly, "F0A": F0A, "F0B": F0B,
                "a": torch.as_tensor(h.a, device=self.dev),
                "b": torch.as_tensor(h.b, device=self.dev),
                "w": torch.as_tensor(h.weight, device=self.dev),
                "amp": torch.as_tensor(amp, device=self.dev),
                "scale": torch.as_tensor(G._scale(g), device=self.dev), "wj": wj,
                "clones": h.clones,
            })  # fmt: skip

    # ------------------------------------------------------------------ CPU side
    def params(self, thetas: np.ndarray, grad: bool):
        """Polynomial coefficients ``C (B, 2, M)``, ``dC (B, D, M)`` and rows (CPU)."""
        Cs, dCs = [], []
        for t in thetas:
            if grad:
                P, dP, rows = self.lik.jacobian(t)
            else:
                P = self.lik.probabilities(t)
                dP, rows = np.zeros((0, P.shape[1])), np.zeros(0, np.int64)
            C, dC = G.poly_coefficients(self.mono[2], len(self.ma), P, dP, rows)
            Cs.append(C)
            dCs.append(dC)
        return np.stack(Cs), np.stack(dCs), rows

    def detection(self, thetas: np.ndarray, grad: bool):
        """Detection term ``-N log q`` and its gradient per vector (CPU, Numba)."""
        lik, m = self.lik, self.lik.model
        val = np.zeros(len(thetas))
        gr = np.zeros((len(thetas), lik.dim))
        if not m.detected:
            return val, gr
        for i, t in enumerate(thetas):
            if grad:
                P, dP, rows = lik.jacobian(t)
            else:
                P = lik.probabilities(t)
                dP, rows = np.zeros((0, P.shape[1])), np.zeros(0, np.int64)
            for h in lik.data:
                q, dq = G._detection_nb(m.offspring, m.n_out, np.asarray(m.capture, float),
                                        np.ascontiguousarray(P), np.ascontiguousarray(dP),
                                        rows, h.generations, m.founder, m.sizes,
                                        m.size_probs)  # fmt: skip
                if q > lik.min_detection:
                    val[i] -= h.clones * np.log(q)
                    if grad:
                        gr[i] -= h.clones * dq / q
                else:
                    val[i] -= h.clones * np.log(lik.min_detection)
        return val, gr

    # ------------------------------------------------------------------ GPU side
    def _powers(self, FA, FB):
        PA, PB = [torch.ones_like(FA)], [torch.ones_like(FB)]
        for _ in range(self.maxp):
            PA.append(PA[-1] * FA)
            PB.append(PB[-1] * FB)
        return PA, PB

    @torch.no_grad()
    def core(self, C: torch.Tensor, grad: bool):
        """Sum over horizons of ``sum w log p`` and its gradient w.r.t. ``C (B, 2, M)``."""
        B = C.shape[0]
        floor = self.lik.floor
        cA = [C[:, 0, m, None, None] for m in range(len(self.ma))]
        cB = [C[:, 1, m, None, None] for m in range(len(self.ma))]
        total = torch.zeros(B, dtype=torch.float64, device=self.dev)
        barC = torch.zeros_like(C)
        for hz in self.hz:
            FA = hz["F0A"].expand(B, -1, -1)
            FB = hz["F0B"].expand(B, -1, -1)
            traj = [(FA, FB)]
            for _ in range(hz["gens"]):
                PA, PB = self._powers(FA, FB)
                nA = 0
                nB = 0
                for m, (a, b) in enumerate(zip(self.ma, self.mb, strict=True)):
                    mo = PA[a] * PB[b]
                    nA = nA + cA[m] * mo
                    nB = nB + cB[m] * mo
                FA, FB = nA, nB
                if grad:
                    traj.append((FA, FB))
            S = self.phi[0] * FA + self.phi[1] * FB
            Gv = sum(pi * S**k for pi, k in zip(self.pis, self.sizes, strict=True))
            p = torch.fft.irfft2(Gv, s=(hz["Lx"], hz["Ly"]))
            pc = p[:, hz["a"], hz["b"]] * hz["amp"]
            ok = pc > floor
            pz = torch.where(ok, pc, torch.full_like(pc, floor))
            total += (hz["w"] * torch.log(pz)).sum(1)
            if not grad:
                continue
            cot = torch.zeros((B, hz["Lx"], hz["Ly"]), dtype=torch.float64, device=self.dev)
            cot[:, hz["a"], hz["b"]] = torch.where(ok, hz["w"] / pz, 0.0) * hz["amp"]
            A = torch.conj(torch.fft.rfft2(cot)) * (hz["wj"] / (hz["Lx"] * hz["Ly"]))
            dG = sum(pi * k * S ** (k - 1) for pi, k in zip(self.pis, self.sizes, strict=True)
                     if k > 0)  # fmt: skip
            lam = A * dG
            lA, lB = lam * self.phi[0], lam * self.phi[1]
            for g in range(hz["gens"] - 1, -1, -1):
                FA, FB = traj[g]
                PA, PB = self._powers(FA, FB)
                if g > 0:
                    fxA = fyA = fxB = fyB = 0
                for m, (a, b) in enumerate(zip(self.ma, self.mb, strict=True)):
                    mo = PA[a] * PB[b]
                    barC[:, 0, m] += (lA * mo).real.sum((1, 2))
                    barC[:, 1, m] += (lB * mo).real.sum((1, 2))
                    if g > 0:
                        if a > 0:
                            gx = a * PA[a - 1] * PB[b]
                            fxA = fxA + cA[m] * gx
                            fxB = fxB + cB[m] * gx
                        if b > 0:
                            gy = b * PA[a] * PB[b - 1]
                            fyA = fyA + cA[m] * gy
                            fyB = fyB + cB[m] * gy
                if g > 0:
                    lA, lB = lA * fxA + lB * fxB, lA * fyA + lB * fyB
        return total, barC

    def value_and_grad(self, thetas: np.ndarray, grad: bool = True, timings: dict | None = None):
        """Log-likelihood ``(B,)`` and gradient ``(B, dim)`` for ``thetas (B, dim)``."""
        t0 = time.perf_counter()
        C, dC, rows = self.params(thetas, grad)
        det_v, det_g = self.detection(thetas, grad)
        t1 = time.perf_counter()
        total, barC = self.core(torch.as_tensor(C, device=self.dev), grad)
        total = total.cpu().numpy()
        barC = barC.cpu().numpy()
        t2 = time.perf_counter()
        ll = total + det_v
        g = None
        if grad:
            g = np.einsum("bdm,bdm->bd", dC, barC[:, rows]) + det_g
        if timings is not None:
            timings["cpu"] = timings.get("cpu", 0.0) + (t1 - t0)
            timings["gpu"] = timings.get("gpu", 0.0) + (t2 - t1)
        return ll, g
