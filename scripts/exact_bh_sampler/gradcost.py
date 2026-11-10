import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from harness import load

for name in sys.argv[1:]:
    lik, *_ = load(name, dt=2.0, threads=1, grid={"factor": 1.1, "margin": 16})
    u = np.zeros(lik.dim)
    lik.log_posterior_u(u)
    lik.log_posterior_u(u, grad=True)
    t = time.perf_counter()
    [lik.log_posterior_u(u) for _ in range(3)]
    e = (time.perf_counter() - t) / 3
    t = time.perf_counter()
    [lik.log_posterior_u(u, grad=True) for _ in range(3)]
    g = (time.perf_counter() - t) / 3
    print(name, f"1 thread: eval {e:.3f}s grad {g:.3f}s ratio {g / e:.1f}", flush=True)
