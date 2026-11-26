"""GPU prototype with fused numba.cuda kernels (one thread per grid point x batch member).

Same algorithm as ``gpu_torch.TorchGW`` (and ``kernel=adjoint``), but the pgf recursion
and its reverse sweep run as two fused kernels in complex128 registers instead of many
memory-bound elementwise torch ops; FFTs, gathers and logs stay in torch. The backward
kernel recomputes the forward recursion per thread (trajectory in local memory) and
reduces the coefficient cotangents per block in shared memory.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from numba import complex128, cuda, float64

from gpu_torch import TorchGW

TPB = 128
MAXG = 32  # max generations (local trajectory arrays)
MAXM = 8  # max monomials
SH_ROWS = 2 * MAXM
TRAJ = MAXG + 1


@cuda.jit(fastmath=False)
def _k_value(C, ma, mb, maxp, gens, zx, zy, F0c, F0x, F0y, phi, sizes, pis, out):
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
    M = ma.shape[0]
    for _ in range(gens):
        A1 = FA
        B1 = FB
        A2 = FA * FA
        B2 = FB * FB
        nA = 0j
        nB = 0j
        for m in range(M):
            a = ma[m]
            bb = mb[m]
            pa = 1.0 + 0j if a == 0 else (A1 if a == 1 else A2)
            pb = 1.0 + 0j if bb == 0 else (B1 if bb == 1 else B2)
            mo = pa * pb
            nA += C[b, 0, m] * mo
            nB += C[b, 1, m] * mo
        FA = nA
        FB = nB
    S = phi[0] * FA + phi[1] * FB
    G = 0j
    for s in range(sizes.shape[0]):
        Sk = 1.0 + 0j
        for _ in range(sizes[s]):
            Sk *= S
        G += pis[s] * Sk
    out[b, i, j] = G


@cuda.jit(fastmath=False)
def _k_adjoint(C, ma, mb, maxp, gens, zx, zy, F0c, F0x, F0y, phi, sizes, pis, Acot, barC):
    t = cuda.grid(1)
    tid = cuda.threadIdx.x
    b = cuda.blockIdx.y
    n = zy.shape[0]
    M = ma.shape[0]
    sh = cuda.shared.array((SH_ROWS, TPB), float64)
    for k in range(2 * M):
        sh[k, tid] = 0.0
    if t < zx.shape[0] * n:
        i = t // n
        j = t - i * n
        x = zx[i]
        y = zy[j]
        trA = cuda.local.array(TRAJ, complex128)
        trB = cuda.local.array(TRAJ, complex128)
        FA = F0c[0] + F0x[0] * x + F0y[0] * y
        FB = F0c[1] + F0x[1] * x + F0y[1] * y
        trA[0] = FA
        trB[0] = FB
        for g in range(gens):
            A2 = FA * FA
            B2 = FB * FB
            nA = 0j
            nB = 0j
            for m in range(M):
                a = ma[m]
                bb = mb[m]
                pa = 1.0 + 0j if a == 0 else (FA if a == 1 else A2)
                pb = 1.0 + 0j if bb == 0 else (FB if bb == 1 else B2)
                mo = pa * pb
                nA += C[b, 0, m] * mo
                nB += C[b, 1, m] * mo
            FA = nA
            FB = nB
            trA[g + 1] = FA
            trB[g + 1] = FB
        S = phi[0] * FA + phi[1] * FB
        dG = 0j
        for s in range(sizes.shape[0]):
            k = sizes[s]
            if k > 0:
                Sk = 1.0 + 0j
                for _ in range(k - 1):
                    Sk *= S
                dG += pis[s] * k * Sk
        lam = Acot[b, i, j] * dG
        lA = lam * phi[0]
        lB = lam * phi[1]
        for g in range(gens - 1, -1, -1):
            FA = trA[g]
            FB = trB[g]
            A2 = FA * FA
            B2 = FB * FB
            fxA = 0j
            fyA = 0j
            fxB = 0j
            fyB = 0j
            for m in range(M):
                a = ma[m]
                bb = mb[m]
                pa = 1.0 + 0j if a == 0 else (FA if a == 1 else A2)
                pb = 1.0 + 0j if bb == 0 else (FB if bb == 1 else B2)
                mo = pa * pb
                sh[m, tid] += (lA * mo).real
                sh[M + m, tid] += (lB * mo).real
                if g > 0:
                    # d/dx x^a with a <= 2: a x^(a-1)
                    if a > 0:
                        da = 1.0 + 0j if a == 1 else 2.0 * FA
                        gx = da * pb
                        fxA += C[b, 0, m] * gx
                        fxB += C[b, 1, m] * gx
                    if bb > 0:
                        db = 1.0 + 0j if bb == 1 else 2.0 * FB
                        gy = pa * db
                        fyA += C[b, 0, m] * gy
                        fyB += C[b, 1, m] * gy
            if g > 0:
                nlA = lA * fxA + lB * fxB
                nlB = lA * fyA + lB * fyB
                lA = nlA
                lB = nlB
    cuda.syncthreads()
    if tid < 2 * M:
        s_ = 0.0
        for r in range(TPB):
            s_ += sh[tid, r]
        k = 0 if tid < M else 1
        cuda.atomic.add(barC, (b, k, tid - k * M), s_)


class NumbaGW(TorchGW):
    """Fused-kernel variant of :class:`gpu_torch.TorchGW` (binary divisions: maxp <= 2)."""

    def __init__(self, lik, device="cuda"):
        super().__init__(lik, device)
        if self.maxp > 2 or len(self.ma) > MAXM:
            raise ValueError("prototype supports at most 2 daughters per outcome")
        if max(hz["gens"] for hz in self.hz) > MAXG:
            raise ValueError(f"prototype supports at most {MAXG} generations")
        m = lik.model
        c0, cx, cy = m.f0
        dev = cuda.to_device
        self.k_args = (dev(np.asarray(self.mono[0])), dev(np.asarray(self.mono[1])),
                       self.maxp)  # fmt: skip
        self.k_obs = (dev(np.asarray(c0, float)), dev(np.asarray(cx, float)),
                      dev(np.asarray(cy, float)), dev(np.asarray(m.founder, float)),
                      dev(np.asarray(m.sizes, np.int64)), dev(np.asarray(m.size_probs, float)))  # fmt: skip
        from bio_sbi.inference import gw_exact as G

        for hz, h in zip(self.hz, lik.data, strict=True):
            zx, zy = G._grid_points(h.grid)
            hz["zx"], hz["zy"] = dev(zx), dev(zy)
            hz["P"] = len(zx) * len(zy)

    @torch.no_grad()
    def core(self, C: torch.Tensor, grad: bool):
        B = C.shape[0]
        floor = self.lik.floor
        Cn = cuda.as_cuda_array(C.contiguous())
        total = torch.zeros(B, dtype=torch.float64, device=self.dev)
        barC = torch.zeros_like(C)
        barCn = cuda.as_cuda_array(barC)
        c0, cx, cy, phi, sizes, pis = self.k_obs
        for hz in self.hz:
            n = hz["zy"].shape[0]
            out = torch.empty((B, hz["Lx"], n), dtype=torch.complex128, device=self.dev)
            blocks = (math.ceil(hz["P"] / TPB), B)
            _k_value[blocks, TPB](Cn, *self.k_args, hz["gens"], hz["zx"], hz["zy"], c0, cx,
                                  cy, phi, sizes, pis, cuda.as_cuda_array(out))  # fmt: skip
            p = torch.fft.irfft2(out, s=(hz["Lx"], hz["Ly"]))
            pc = p[:, hz["a"], hz["b"]] * hz["amp"]
            ok = pc > floor
            pz = torch.where(ok, pc, torch.full_like(pc, floor))
            total += (hz["w"] * torch.log(pz)).sum(1)
            if not grad:
                continue
            cot = torch.zeros((B, hz["Lx"], hz["Ly"]), dtype=torch.float64, device=self.dev)
            cot[:, hz["a"], hz["b"]] = torch.where(ok, hz["w"] / pz, 0.0) * hz["amp"]
            A = torch.conj(torch.fft.rfft2(cot)) * (hz["wj"] / (hz["Lx"] * hz["Ly"]))
            A = A.contiguous()
            _k_adjoint[blocks, TPB](Cn, *self.k_args, hz["gens"], hz["zx"], hz["zy"], c0, cx,
                                    cy, phi, sizes, pis, cuda.as_cuda_array(A), barCn)  # fmt: skip
        return total, barC
