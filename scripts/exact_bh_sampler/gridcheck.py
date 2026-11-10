"""lp error and cost of cheaper settings (dt, grid factor/margin) at posterior draws."""

import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from harness import load

name, npz = sys.argv[1], sys.argv[2]
z = np.load(npz)
P = z["u"]
rng = np.random.default_rng(0)
U = P[rng.choice(len(P), 32, replace=False)]
ref_lik, *_ = load(name, dt=1.0, threads=16)
ref = ref_lik.log_posterior_batch(U)
ref_lik.close()
for dt, g in (
    (2.0, {}),
    (3.0, {}),
    (4.0, {}),
    (2.0, {"factor": 1.1, "margin": 16}),
    (3.0, {"factor": 1.1, "margin": 16}),
    (3.0, {"factor": 1.0, "margin": 8}),
):
    lik, *_ = load(name, dt=dt, threads=16, grid=g)
    lik.log_posterior_batch(U[:16])
    t = time.perf_counter()
    lp = lik.log_posterior_batch(U)
    s = (time.perf_counter() - t) / len(U)
    print(
        f"dt {dt} grid {g} {lik.grid.Lx}x{lik.grid.Ly}: {s:.3f} s/eval, max |dlp| {np.abs(lp - ref).max():.2e}",
        flush=True,
    )
    lik.close()
