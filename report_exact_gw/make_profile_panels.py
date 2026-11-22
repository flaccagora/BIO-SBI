"""Profile-likelihood panels of B.to_A for Section 6 of report_exact_gw.

Usage (from the repository root)::

    uv run python report_exact_gw/make_profile_panels.py

Reads the profiles stored by ``scripts/xg2_mle_vs_posterior/profile_runs.py`` in
``docs/xg2_mle_vs_posterior/profiles/<gt>_s<seed>_n<n>.npz`` and writes
``report_exact_gw/figures/profile_panels.pdf``: for each run, the profile log likelihood of
B.to_A and the log marginal posterior (flat prior), and below it B's division mass at the
profile optimum (what the other parameters do along the ridge).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "xg2_mle_vs_posterior"))
from analyse_profiles import kde_logpdf  # noqa: E402

PROF = ROOT / "docs" / "xg2_mle_vs_posterior" / "profiles"
FIG = ROOT / "report_exact_gw" / "figures"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK2, MUTED = "#52514e", "#898781"
RUNS = [
    ("T1_ref_s0_n128", "T1 (B→A 0.8, B never divides), n = 128"),
    ("T1_ref_s0_n512", "T1, n = 512"),
    ("T4_slowA_s0_n512", "T4 (slow A, B→A 0.8), n = 512"),
    ("T2_toA05_s0_n512", "T2 (B→A 0.5, B never divides), n = 512"),
    ("T3_Bdiv_s0_n512", "T3 (B→A 0.6, B divides), n = 512"),
    ("P3_s0_n512", "P3 (prior draw, B→A 0.27), n = 512"),
]


def main() -> None:
    fig, axes = plt.subplots(4, 3, figsize=(12.5, 9.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1, 2.2, 1]})  # fmt: skip
    for k, (stem, title) in enumerate(RUNS):
        z = np.load(PROF / f"{stem}.npz")
        names = list(z["names"])
        ia = names.index("B.to_A")
        bidx = [names.index(n) for n in ("B.div_AA", "B.div_AB", "B.div_BB")]
        grid, prof = z["grid"], z["prof_ll"]
        dprof = prof - max(float(z["ll_mle"]), prof.max())
        x = z["post"][:, ia]
        logp = kde_logpdf(x, grid)
        logp -= logp.max()
        r, c = 2 * (k // 3), k % 3
        a, b = axes[r, c], axes[r + 1, c]
        a.plot(grid, dprof, "-o", ms=3, color=BLUE, label="profile log likelihood")
        a.plot(grid, logp, "-", color=ORANGE, label="log marginal posterior (flat prior)")
        a.axhline(-1.92, color=MUTED, lw=0.8, ls=":")
        truth, mle = float(z["truth"][ia]), float(z["mle_phys"][ia])
        a.axvline(truth, color="k", lw=1.2, label="truth")
        a.axvline(mle, color=BLUE, lw=1, ls="--", label="MLE")
        a.axvline(x.mean(), color=ORANGE, lw=1, ls="-.", label="posterior mean")
        a.set_ylim(max(-12, min(dprof.min(), logp.min()) - 0.5), 0.8)
        a.set_title(title, fontsize=9.5)
        if c == 0:
            a.set_ylabel("nats (max = 0)")
        b.plot(grid, z["prof_phys"][:, bidx].sum(1), "-", color=YELLOW, label="B division mass")
        b.plot(grid, z["prof_phys"][:, names.index("B.death")], "-", color=INK2, lw=1,
               label="B dies")  # fmt: skip
        b.axvline(truth, color="k", lw=1.2)
        b.set_ylim(-0.03, 0.85)
        if c == 0:
            b.set_ylabel("at profile\noptimum", fontsize=8.5)
        if r == 2:
            b.set_xlabel("B→A fixed at c")
        for ax in (a, b):
            ax.grid(alpha=0.3, lw=0.5)
            ax.tick_params(labelsize=8)
    axes[0, 0].legend(fontsize=7.5, loc="lower left")
    axes[1, 0].legend(fontsize=7.5, loc="upper right")
    fig.tight_layout(h_pad=0.6)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "profile_panels.pdf")
    fig.savefig(FIG / "profile_panels.png", dpi=110)
    print("wrote", FIG / "profile_panels.pdf")


if __name__ == "__main__":
    main()
