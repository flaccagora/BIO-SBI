"""Per-call time of the exact GW likelihood: kernel=loop vs kernel=adjoint.

Design: two_type (10 free coordinates), reference truth, detected clones; generations
{4,6,8,10} and {4,6,10}; 128 and 1024 clones per endpoint. ``threads=1``: one call at a
time; ``threads=2``: two Python threads calling concurrently (two NUTS chains), reported
as wall time per call (= 1 / throughput) and the latency of one call.

    OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 uv run python scripts/gw_speedup/bench_kernels.py \
        > scripts/gw_speedup/bench_kernels.txt
"""

import copy
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from profile_parts import make

REPS = 20
ROUNDS = 5


def per_call(fn, threads):
    fn()
    if threads == 1:
        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return np.median(ts) * 1e3, np.median(ts) * 1e3

    lat = []

    def work(_):
        for _ in range(REPS):
            t0 = time.perf_counter()
            fn()
            lat.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(work, range(threads)))
    wall = time.perf_counter() - t0
    return wall / (threads * REPS) * 1e3, np.median(lat) * 1e3


def main():
    print("gens | clones | grids | threads | op | loop ms/call | adjoint ms/call | speed-up | "
          "loop latency | adjoint latency")
    for gens in [(4, 6, 8, 10), (4, 6, 10)]:
        for clones in (128, 1024):
            lik, th, _ = make("two_type", gens, clones)
            adj = copy.copy(lik)
            adj.kernel = "adjoint"
            grids = " ".join(f"{h.grid.Lx}x{h.grid.Ly}" for h in lik.data)
            for threads in (1, 2):
                for op, grad in (("objective", False), ("gradient", True)):
                    # alternate blocks (machine load drifts): median over ROUNDS blocks each
                    ra, rb = [], []
                    for _ in range(ROUNDS):
                        ra.append(per_call(lambda grad=grad: lik.log_likelihood(th, grad), threads))
                        rb.append(per_call(lambda grad=grad: adj.log_likelihood(th, grad), threads))
                    a, la = np.median(ra, axis=0)
                    b, lb = np.median(rb, axis=0)
                    print(f"{gens} | {clones} | {grids} | {threads} | {op} | {a:.2f} | "
                          f"{b:.2f} | {a / b:.2f}x | {la:.2f} | {lb:.2f}", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
