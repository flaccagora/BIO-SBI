"""Validate and benchmark the batched GPU likelihood (gpu_torch.TorchGW) vs the CPU.

    OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 uv run python scripts/gw_speedup/bench_gpu.py \
        > scripts/gw_speedup/bench_gpu.txt
"""

import copy
import sys
import time

import numpy as np
import torch
from scipy.special import expit, logit

from gpu_torch import TorchGW
from profile_parts import make

torch.set_num_threads(2)
DESIGNS = [("two_type", (4, 6, 8, 10), 128), ("two_type", (4, 6, 8, 10), 1024),
           ("two_type", (4, 6, 10), 1024), ("reversible_ab", (4, 6), 128)]  # fmt: skip
BATCHES = [1, 4, 16, 64, 256]


def sync():
    torch.cuda.synchronize()


def main(kind="torch"):
    rng = np.random.default_rng(0)
    print("design | B | GPU total ms/eval | of which CPU-side ms/eval | GPU-core ms/eval | "
          "evals/s | CPU adjoint 2 thr ms/eval | CPU loop 2 thr ms/eval | speed-up vs "
          "adjoint | vs loop")  # fmt: skip
    for name, gens, clones in DESIGNS:
        lik, th, _ = make(name, gens, clones)
        adj = copy.copy(lik)
        adj.kernel = "adjoint"
        if kind == "torch":
            gpu = TorchGW(lik)
        else:
            from gpu_numba import NumbaGW

            gpu = NumbaGW(lik)
        # validation on well-conditioned points
        pts = expit(logit(np.clip(th, 1e-6, 1 - 1e-6))[None]
                    + rng.normal(0, 0.15, size=(16, lik.dim)))  # fmt: skip
        ll, g = gpu.value_and_grad(pts)
        dv, dg = [], []
        for i, t in enumerate(pts):
            v0, g0 = lik.log_likelihood(t)
            dv.append(abs(ll[i] - v0) / abs(v0))
            dg.append(np.abs(g[i] - g0).max() / np.abs(g0).max())
        print(f"# {name} {gens} x {clones}: grids "
              f"{[(h.grid.Lx, h.grid.Ly) for h in lik.data]}; validation vs CPU loop on 16 "
              f"points: value rel diff max {max(dv):.1e}, grad rel diff max {max(dg):.1e}")
        # CPU reference throughput: 2 threads, gradient calls
        from bench_kernels import per_call

        cpu_adj = per_call(lambda: adj.log_likelihood(th), 2)[0]
        cpu_loop = per_call(lambda: lik.log_likelihood(th), 2)[0]
        for B in BATCHES:
            thetas = expit(logit(np.clip(th, 1e-6, 1 - 1e-6))[None]
                           + rng.normal(0, 0.1, size=(B, lik.dim)))  # fmt: skip
            try:
                gpu.value_and_grad(thetas)
                sync()
            except torch.OutOfMemoryError:
                print(f"{name} {gens}x{clones} | {B} | out of memory")
                torch.cuda.empty_cache()
                continue
            reps = max(3, 64 // B)
            tm = {}
            t0 = time.perf_counter()
            for _ in range(reps):
                gpu.value_and_grad(thetas, timings=tm)
            sync()
            per = (time.perf_counter() - t0) / (reps * B) * 1e3
            cpu_side = tm["cpu"] / (reps * B) * 1e3
            print(f"{name} {gens}x{clones} | {B} | {per:.3f} | {cpu_side:.3f} | "
                  f"{tm['gpu'] / (reps * B) * 1e3:.3f} | {1e3 / per:.0f} | {cpu_adj:.2f} | "
                  f"{cpu_loop:.2f} | {cpu_adj / per:.2f}x | {cpu_loop / per:.2f}x",
                  flush=True)  # fmt: skip
        del gpu
        torch.cuda.empty_cache()
    print("peak GPU memory GiB", torch.cuda.max_memory_allocated() / 2**30)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "torch")
