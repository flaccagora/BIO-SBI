"""Check the numpy simulator of the misspecification study against the pipeline.

1. Correctly specified case, truths T1_ref, T3_Bdiv, T5_C7, P1: 20,000 detected clones per
   horizon from (a) this study's simulator and (b) the pipeline's exact detection-conditioned
   sampler (``GaltonWatson.simulate_detected``); both are tested against the *exact*
   detection-conditioned law (binned G^2, no fitted parameters) and their summaries compared.
2. Scenarios whose law the exact likelihood can represent (founder mixture + doublets,
   cell loss, symmetric mis-typing): this simulator against the exact law of the extended
   model (founder / size / capture / confusion settings).
3. Clone heterogeneity (kappa = 10): this simulator against the pipeline's random-effects
   law (``law.clone_heterogeneity``, exact detection conditioning by rejection on the clone
   matrices): two-sample chi-square on the same bins.

Writes ``validation.csv`` / ``validation.md`` to the results folder and prints them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

from bio_sbi.summaries import Cohort  # noqa: E402

N = 20_000


def gof_known(lik, P):
    g = C.gof(lik, P)
    return g["G2"], g["df_upper"], g["p_upper"]


def summ(cohorts):
    out = {}
    for c in cohorts:
        a, b = c.counts[:, 0], c.counts[:, 1]
        h = int(c.horizon)
        out[f"meanA{h}"] = a.mean()
        out[f"meanB{h}"] = b.mean()
        out[f"pA0_{h}"] = np.mean(a == 0)
    return out


def pipeline_detected(P, n, seed, law_extra=None, phys_extra=None):
    from bio_sbi.rng import rng_for
    from bio_sbi.simulators import GaltonWatson

    conf = yaml.safe_load((C.ROOT / "src/bio_sbi/conf/law/two_type.yaml").read_text())
    conf.update(law_extra or {})
    s = GaltonWatson(conf)
    mp = C.dl.P_to_dict(P)
    mp.update(phys_extra or {})
    phys = s.physical_from_mapping(mp)
    out = []
    for i, h in enumerate(C.HORIZONS):
        x = s.simulate_detected(phys, h, n, np.ones(2), rng_for(seed, "validate", i))
        assert np.all(x >= 0)
        out.append(Cohort(float(h), x.astype(np.int64)))
    return out


def two_sample(c1, c2):
    """Chi-square two-sample test on the GOF bins of the pooled data (per horizon)."""
    chi, df = 0.0, 0
    for x, y in zip(c1, c2, strict=True):
        allc = np.concatenate([x.counts, y.counts])
        tot = allc.sum(1)
        nb = np.unique(np.quantile(tot, np.linspace(0, 1, 7)[1:-1])) + 0.5
        kx = C._classes(x.counts[:, 0], x.counts[:, 1], nb)
        ky = C._classes(y.counts[:, 0], y.counts[:, 1], nb)
        m = max(kx.max(), ky.max()) + 1
        tab = np.array([np.bincount(kx, minlength=m), np.bincount(ky, minlength=m)])
        tab = tab[:, tab.sum(0) >= 10]
        r = stats.chi2_contingency(tab)
        chi += r.statistic
        df += r.dof
    return chi, df, float(stats.chi2.sf(chi, df))


def main():
    rows = []
    for gt in ("T1_ref", "T3_Bdiv", "T5_C7", "P1"):
        P = C.TRUTHS[gt]
        mine = C.simulate(P, N, np.random.default_rng([7, C.GTS.index(gt)]))
        pipe = pipeline_detected(P, N, 11 + C.GTS.index(gt))
        for name, data in (("numpy", mine), ("pipeline", pipe)):
            g2, df, p = gof_known(C.make_lik(data), P)
            rows.append({"check": "base vs exact law", "truth": gt, "sampler": name,
                         "stat": g2, "df": df, "p": p, **summ(data)})  # fmt: skip
        chi, df, p = two_sample(mine, pipe)
        rows.append({"check": "base numpy vs pipeline", "truth": gt, "sampler": "2-sample",
                     "stat": chi, "df": df, "p": p})  # fmt: skip
        # extended-law scenarios
        for scen, kw, lk in (
            ("D15mix", {"founder_A": 0.7, "doublet": 0.15},
             {"founder_A": 0.7, "size_probs": (0.85, 0.15)}),
            ("LOSS80", {"capture": 0.8}, {"capture": 0.8}),
            ("MTsym2", {"confusion": C.dl.confusion_matrix(0.02, "symmetric")},
             {"confusion": C.dl.confusion_matrix(0.02, "symmetric")}),
        ):  # fmt: skip
            data = C.simulate(P, N, np.random.default_rng([8, C.GTS.index(gt), len(scen)]), **kw)
            g2, df, p = gof_known(C.make_lik(data, **lk), P)
            rows.append({"check": f"{scen} vs exact extended law", "truth": gt,
                         "sampler": "numpy", "stat": g2, "df": df, "p": p})  # fmt: skip
        if gt in ("T1_ref", "T5_C7"):
            # pipeline founder mixture + doublets, against the same exact law
            pf = pipeline_detected(P, N, 21, {"founder_sizes": [1, 2],
                                              "fixed": {"founder": {"A": 0.7, "B": 0.3},
                                                        "founder_size": {"1": 0.85, "2": 0.15}}},
                                   {"founder.A": 0.7, "founder.B": 0.3,
                                    "founder_size.1": 0.85, "founder_size.2": 0.15})  # fmt: skip
            g2, df, p = gof_known(C.make_lik(pf, founder_A=0.7, size_probs=(0.85, 0.15)), P)
            rows.append({"check": "D15mix vs exact extended law", "truth": gt,
                         "sampler": "pipeline", "stat": g2, "df": df, "p": p})  # fmt: skip
            # heterogeneity
            mh = C.simulate(P, N // 2, np.random.default_rng([9, C.GTS.index(gt)]), kappa=10.0)
            ph = pipeline_detected(P, N // 2, 31,
                                   {"clone_heterogeneity": {"kappa_bounds": [1.0, 1000.0]},
                                    "fixed": {"founder": {"A": 1.0}, "clone": {"kappa": 10.0}}},
                                   {"clone.kappa": 10.0})  # fmt: skip
            chi, df, p = two_sample(mh, ph)
            rows.append({"check": "HET10 numpy vs pipeline", "truth": gt, "sampler": "2-sample",
                         "stat": chi, "df": df, "p": p, **summ(mh)})  # fmt: skip
            rows.append({"check": "HET10 pipeline", "truth": gt, "sampler": "pipeline",
                         **summ(ph)})  # fmt: skip
    df = pd.DataFrame(rows)
    C.OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.OUT / "validation.csv", index=False)
    cols = ["check", "truth", "sampler", "stat", "df", "p", "meanA10", "meanB10", "pA0_10"]
    md = df[[c for c in cols if c in df]].to_markdown(index=False, floatfmt=".3g")
    (C.OUT / "validation.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
