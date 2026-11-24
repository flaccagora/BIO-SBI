"""Slow- and fast-mixing parameter sets of the composition-convergence study (12 probabilities,
``common.NAMES`` order). Same numbers as ``truths.sh`` (the NUTS runs).

* S0: rare dedifferentiation (B->A 0.01), A and B self-renew equally (a ~ d) - slowest;
* S1: as S0 with B->A 0.02;
* S3: B->A 0.05, B self-renews a little more than A;
* F1: S1's A row, B row with frequent return to A (fast mixing, similar growth).
"""

import numpy as np

SETS = {
    "S0": [.08, .10, .08, .58, .10, .06, .20, .01, .19, .005, .01, .585],
    "S1": [.08, .10, .10, .55, .12, .05, .20, .02, .18, .01, .02, .57],
    "S3": [.08, .10, .10, .52, .12, .08, .15, .05, .12, .02, .04, .62],
    "F1": [.08, .10, .10, .55, .12, .05, .20, .30, .10, .10, .10, .20],
}  # fmt: skip
SETS = {k: np.array(v, float) for k, v in SETS.items()}
SLOW = ["S0", "S1", "S3"]
FAST = ["F1"]
