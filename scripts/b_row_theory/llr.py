"""E0 at 4096: exact log-likelihood of theta* vs the NPE posterior mean on the stored reference data."""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gwexact import *
order = NAMES
# gradient check by finite differences
T = tangent_basis()
p, g = law_and_grads(THETA_STAR + 0, 6, T[[8]], L=65)
h = 1e-6
fd = (law(THETA_STAR + h*T[8], 6, 65) - law(THETA_STAR - h*T[8], 6, 65)) / (2*h)
print("grad check max err", np.abs(fd - g[0]).max(), "scale", np.abs(g[0]).max())

H = (4, 6, 10)
# results/ is git-ignored; point BIO_SBI_RESULTS at the checkout that holds the E4 runs.
RESULTS = os.environ.get("BIO_SBI_RESULTS",
                         os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results"))
laws_star = {n: law(THETA_STAR, n) for n in H}
for tag in ("gw_xxl_two_type_20k", "gw_xxl_two_type_5k"):
    for d in sorted(glob.glob(f"{RESULTS}/e4/{tag}/[0-9]_*")):
        rec = pd.read_csv(f"{d}/recovery.csv").set_index("parameter")
        th = rec.loc[order, "posterior_mean"].to_numpy().copy()
        th[:6] /= th[:6].sum(); th[6:] /= th[6:].sum()
        ref = np.load(f"{d}/reference.npz")
        llr = 0; kl = 0
        for n in H:
            c = ref[f"counts_h{n}"]
            ls, _ = cond_logp(laws_star[n]); pa = law(th, n); la, _ = cond_logp(pa)
            llr += (ls[c[:,0], c[:,1]] - la[c[:,0], c[:,1]]).sum()
            kl += kl_cond(laws_star[n], pa)
        print(f"{tag} seed{d.split('/')[-1][0]}  B.to_A={th[8]:.2f} Bdiv={th[9:].sum():.2f}  "
              f"loglik(θ*)-loglik(θ̂) on data = {llr:8.1f}   KL/clone(sum over 3 endpoints)={kl:.4f}  4096*KL={4096*kl:.0f}")
