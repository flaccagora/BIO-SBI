"""Compare posterior draws of a sampler run vs a reference (MC errors from chain ESS).
python compare.py NAME ref.npz run.npz
"""

import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from harness import load

from bio_sbi.inference.smc import ess_bulk

name, ref, run = sys.argv[1:4]
lik, *_ = load(name, threads=1)
sp = lik.space
free = [i for i, n in enumerate(sp.names) if n in set(sp.free_names)]


def physical_chains(path):
    z = np.load(path)
    U = z["chains"] if "chains" in z.files else z["u"]
    if U.ndim == 2:
        U = U[None]
    C, L, d = U.shape
    phys = sp.decode(lik.theta_from_u(U.reshape(-1, d))).reshape(C, L, -1)
    return phys


R, S = physical_chains(ref), physical_chains(run)
print(
    f"{'param':16s} {'ref mean':>8s} {'run mean':>8s} {'z':>6s} {'ref sd':>7s} {'run sd':>7s} {'sd z':>6s} "
    f"{'q05 ref/run':>15s} {'q50 ref/run':>15s} {'q95 ref/run':>15s} {'ESS r/s':>10s}"
)
zs, zsd = [], []
for i in free:
    r, s = R[:, :, i], S[:, :, i]
    er, es = ess_bulk(r), ess_bulk(s)
    rf, sf = r.ravel(), s.ravel()
    se = np.sqrt(rf.var() / er + sf.var() / es)
    z = (sf.mean() - rf.mean()) / se
    # sd: relative MC error ~ sqrt((kurtosis-1)/(4 ESS)) ~ 1/sqrt(2 ESS) for near-Gaussian
    k = np.mean((rf - rf.mean()) ** 4) / rf.var() ** 2
    se_sd = np.sqrt((k - 1) / 4 * (rf.std() ** 2 / er + sf.std() ** 2 / es))
    z2 = (sf.std() - rf.std()) / se_sd
    zs.append(z)
    zsd.append(z2)
    q = lambda x, p: np.quantile(x, p)
    print(
        f"{sp.names[i]:16s} {rf.mean():8.4f} {sf.mean():8.4f} {z:6.2f} {rf.std():7.4f} {sf.std():7.4f} {z2:6.2f} "
        f"{q(rf, 0.05):7.4f}/{q(sf, 0.05):7.4f} {q(rf, 0.5):7.4f}/{q(sf, 0.5):7.4f} {q(rf, 0.95):7.4f}/{q(sf, 0.95):7.4f} {er:5.0f}/{es:4.0f}"
    )
print(
    "max |z| mean", np.max(np.abs(zs)), " max |z| sd", np.max(np.abs(zsd)), " (", len(zs), "params)"
)
