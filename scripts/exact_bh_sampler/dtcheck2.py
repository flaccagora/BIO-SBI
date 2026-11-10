import glob
import json
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from harness import RUNS, load

m = json.load(open(glob.glob(RUNS["rev0"])[0] + "/metrics.json"))["exact_bh"]
liks = {dt: load("rev0", dt=dt, threads=4)[0] for dt in (0.5, 1.0, 2.0, 3.0, 4.0)}
l1 = liks[1.0]
u0 = l1.u_from_theta(np.array(m["mode_theta"]))
sd = np.array(m["laplace_sd_u"])
rng = np.random.default_rng(1)
for k in range(5):
    u = u0 + (rng.normal(size=len(u0)) * sd * 1.5 if k else 0)
    ref = liks[0.5].log_posterior_u(u)
    print(
        f"lp={ref:.4f}",
        " ".join(
            f"dt{dt:g}:{liks[dt].log_posterior_u(u) - ref:+.1e}" for dt in (1.0, 2.0, 3.0, 4.0)
        ),
        flush=True,
    )
