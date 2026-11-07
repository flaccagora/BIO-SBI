import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gwexact import *
T = tangent_basis()   # 10 x 12, coords: A.{persist,to_B,div_AA,div_AB,div_BB}, B.{persist,to_A,div_AA,div_AB,div_BB} (death absorbs)
lab = ["A.persist","A.to_B","A.div_AA","A.div_AB","A.div_BB","B.persist","B.to_A","B.div_AA","B.div_AB","B.div_BB"]
Ls = {4:17, 6:65, 8:257, 10:384, 12:1024}
I_n = {}
for n in (4,6,8,10,12):
    p, g = law_and_grads(THETA_STAR, n, T, L=Ls[n])
    I, miss = conditional_fisher(p, g)
    I_n[n] = I
    print(f"n={n}: missed mass {miss:.1e}, eig min/max {np.linalg.eigvalsh(I).min():.3e} / {np.linalg.eigvalsh(I).max():.3e}")
np.savez("fisher.npz", **{f"I{n}": I for n, I in I_n.items()})
# functional: total B division mass delta = div_AA+div_AB+div_BB ; gradient in these coords
gd = np.zeros(10); gd[7:] = 1
gt = np.zeros(10); gt[6] = 1  # B.to_A
designs = {"E0 {4,8,12}x32": {4:32,8:32,12:32}, "real-like {6:165,10:101}": {6:165,10:101},
           "E0-large {4,6,10}x4096": {4:4096,6:4096,10:4096}}
for name, des in designs.items():
    I = sum(N*I_n[n] for n, N in des.items())
    C = np.linalg.inv(I)
    print(f"\n{name}: cond(I)={np.linalg.cond(I):.2e}")
    print("  CR sd B.to_A     :", np.sqrt(gt@C@gt).round(4))
    print("  CR sd B-div mass :", np.sqrt(gd@C@gd).round(4))
    sd = np.sqrt(np.diag(C)); print("  CR sd per coord  :", dict(zip(lab, sd.round(3))))
    w, V = np.linalg.eigh(I)
    print("  weakest eigvec (eig %.3g, sd %.3f):" % (w[0], 1/np.sqrt(w[0])), dict(zip(lab, V[:,0].round(2))))
# per-clone information per endpoint, for the two functionals
for n in (4,6,8,10,12):
    try:
        C = np.linalg.inv(I_n[n]); print(f"single endpoint n={n}: per-clone CR var  to_A {gt@C@gt:.3g}, Bdiv {gd@C@gd:.3g}, cond {np.linalg.cond(I_n[n]):.2e}")
    except Exception as e: print(n, e)
