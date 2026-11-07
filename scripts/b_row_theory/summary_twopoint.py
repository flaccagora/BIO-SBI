"""Does the `moments` summary separate theta* from the NPE posterior mean theta_hat?

theta_hat = posterior mean of the E0-at-4096 run (GW, two_type, 20k sims, seed 0).
R datasets per side at {4,6,10} x N clones; report the cross-validated accuracy of a
regularised logistic regression on the summaries, and the exact-likelihood KL for
comparison (expected log LR per dataset = N * sum_n KL_n).
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gwexact import THETA_STAR, OFF, law, kl_cond
from bio_sbi.simulators.galton_watson import gw_forward
from bio_sbi.summaries import CohortMoments, Cohort
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

THETA_HAT = np.array([.0888, .1531, .0164, .4711, .0578, .2128,      # A (posterior means)
                      .1475, .0406, .3240, .2122, .1116, .1641])     # B: death, persist, to_A, AA, AB, BB
THETA_HAT[:6] /= THETA_HAT[:6].sum(); THETA_HAT[6:] /= THETA_HAT[6:].sum()
offspring = np.array([[OFF[k][o] for o in range(6)] for k in (0, 1)], dtype=np.int64)
n_out = np.array([6, 6], dtype=np.int64)
founder = np.array([1.0, 0.0]); sizes = np.array([1], dtype=np.int64); sp = np.array([1.0])
summ = CohortMoments()


def dataset(P, design, rng):
    out = []
    for n, N in design.items():
        c = gw_forward(P.reshape(2, 6), n_out, offspring, founder, sizes, sp, n, int(N * 1.1) + 50, rng, 10**6)
        out.append(Cohort(n, c[c.sum(1) > 0][:N]))
    return out


rng = np.random.default_rng(0)
H = (4, 6, 10); Ls = {4: 17, 6: 65, 10: 384}
kl = sum(kl_cond(law(THETA_STAR, n, Ls[n]), law(THETA_HAT, n, Ls[n])) for n in H)
for N, R in ((165, 400), (4096, 300)):
    des = {n: N for n in H}
    X = np.array([summ(dataset(P, des, rng)) for P in (THETA_STAR, THETA_HAT) for _ in range(R)])
    y = np.repeat([0, 1], R)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
    acc = cross_val_score(clf, X, y, cv=5).mean()
    d = X[y == 1].mean(0) - X[y == 0].mean(0); s = X.std(0) + 1e-12
    top = np.argsort(-np.abs(d / s))[:5]
    stats = ["mean", "sd", "q.1", "q.25", "q.5", "q.75", "q.9"]
    names = [f"h{n}.{t}.{s}" for n in H for t in ("A", "B", "total")
             for s in stats + (["present"] if t != "total" else ["empty"])]
    names = [nm for n in H for nm in [x for x in names if x.startswith(f"h{n}.")] + [f"h{n}.corr_AB"]]
    print(f"N={N}/endpoint: summary classifier CV accuracy {acc:.3f}; exact expected log-LR per dataset "
          f"{N * kl:.1f} nats; top shifted features (effect size): "
          + ", ".join(f"{names[i]} {d[i] / s[i]:+.2f}" for i in top), flush=True)
