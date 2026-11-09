import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gwexact import *
T = tangent_basis()
Ls = {2:5, 3:9, 4:17, 5:33, 6:65, 7:129, 8:257, 9:384, 10:384, 12:1024}
I = dict(np.load("fisher.npz")); I = {int(k[1:]): v for k, v in I.items()}
for n in (2, 3, 5, 7, 9):
    p, g = law_and_grads(THETA_STAR, n, T, L=Ls[n]); I[n] = conditional_fisher(p, g)[0]
gt = np.zeros(10); gt[6] = 1; gd = np.zeros(10); gd[7:] = 1
print(" n | eig_min      cond     | per-clone CR var: B.to_A  B-div | clones for sd 0.05 (to_A)")
for n in sorted(I):
    w = np.linalg.eigvalsh(I[n]); C = np.linalg.pinv(I[n]) if w[0] < 1e-10 else np.linalg.inv(I[n])
    print(f"{n:2d} | {w[0]:.3e} {w[-1]/max(w[0],1e-300):.2e} | {gt@C@gt:9.3g} {gd@C@gd:9.3g} | {gt@C@gt/0.0025:9.0f}")
np.savez("fisher_all.npz", **{f"I{n}": v for n, v in I.items()})
def cr(des):
    J = sum(N * I[n] for n, N in des.items()); C = np.linalg.inv(J); return np.sqrt(gt@C@gt), np.sqrt(gd@C@gd)
print("\nDesign (clones per endpoint)              CR sd B.to_A   CR sd B-div")
for name, des in [("real-like {6:165,10:101}", {6:165, 10:101}),
                  ("E0 {4,8,12}x32", {4:32, 8:32, 12:32}),
                  ("{4,6,10}x512", {4:512, 6:512, 10:512}),
                  ("{4,6,10}x4096 (E0-large)", {4:4096, 6:4096, 10:4096}),
                  ("{3,6,10}x{89,89,88} (266 total)", {3:89, 6:89, 10:88}),
                  ("{2,4,6,10}x{67,67,66,66}", {2:67, 4:67, 6:66, 10:66}),
                  ("{4}x266", {4:266}), ("{3}x266", {3:266}), ("{2,3}x133", {2:133, 3:133}),
                  ("{3,4}x1000", {3:1000, 4:1000})]:
    a, b = cr(des); print(f"{name:40s} {a:8.3f}     {b:8.3f}")
