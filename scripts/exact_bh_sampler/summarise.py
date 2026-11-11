"""ESS per 1k evaluations and agreement with the reference: python summarise.py NAME REF run1.json ..."""

import json
import sys

import numpy as np
from harness import load

from bio_sbi.inference.smc import ess_bulk, ess_tail, nested_rhat

name, ref = sys.argv[1], sys.argv[2]
lik, *_ = load(name, threads=1)
sp = lik.space
free = [i for i, n in enumerate(sp.names) if n in set(sp.free_names)]


def phys(U):
    """Physical parameters of chains ``(C, L, d)`` in u coordinates."""
    C, L, d = U.shape
    return sp.decode(lik.theta_from_u(U.reshape(-1, d))).reshape(C, L, -1)


_z = np.load(ref) if ref != "-" else None
R = phys(_z["chains"] if "chains" in _z.files else _z["u"]) if _z is not None else None
for f in sys.argv[3:]:
    j = json.load(open(f))
    z = np.load(f.replace(".json", ".npz"))
    U = z["chains"]
    C, L, d = U.shape
    ess = [ess_bulk(U[:, :, i]) for i in range(d)]
    et = [ess_tail(U[:, :, i]) for i in range(d)]
    nr = max(nested_rhat(U[:, :, i], 4) for i in range(d))
    ev = j["likelihood_evals"]
    # final-stage evaluations (after the tempering stages)
    st = j["smc"]["stage_diagnostics"]
    fin_ev = ev - st[-2]["evals"] if len(st) > 1 else ev
    line = (
        f"{f.split('/')[-1]:28s} evals {ev:6d} (final {fin_ev:6d}) wall {j['wall']:6.0f}s "
        f"minESS {min(ess):5.0f} ({sp.theta_names[int(np.argmin(ess))]}) tail {min(et):4.0f} "
        f"nRhat {nr:.4f} | ESS/1k evals {1000 * min(ess) / ev:5.2f} (final-only "
        f"{1000 * min(ess) / fin_ev:5.2f}) acc {st[-1]['acceptance_rw']:.2f}/{st[-1]['acceptance_imh']:.2f}"
    )
    print(line)
    if R is not None:
        S = phys(U)
        zs, zsd = [], []
        for i in free:
            r, s = R[:, :, i], S[:, :, i]
            er, es = ess_bulk(r), ess_bulk(s)
            rf, sf = r.ravel(), s.ravel()
            se = np.sqrt(rf.var() / er + sf.var() / es)
            k = np.mean((rf - rf.mean()) ** 4) / rf.var() ** 2
            se_sd = np.sqrt((k - 1) / 4 * (rf.std() ** 2 / er + sf.std() ** 2 / es))
            zs.append((sf.mean() - rf.mean()) / se)
            zsd.append((sf.std() - rf.std()) / se_sd)
        print(
            f"    vs reference: max |z| mean {np.max(np.abs(zs)):.2f}, max |z| sd {np.max(np.abs(zsd)):.2f}"
            f"  (mean z {' '.join(f'{v:+.1f}' for v in zs)}; sd z {' '.join(f'{v:+.1f}' for v in zsd)})"
        )
