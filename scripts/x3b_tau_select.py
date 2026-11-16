"""X3b: choose the GW hours per generation tau by marginal likelihood on one dataset.

Runs ``bio-sbi`` (exact GW, ``inference=exact_gw``; its log evidence is in
``metrics.json``) once per distinct generation tuple of the tau grid -- taus that round
every horizon to the same generations (``g = round_half_up(h / tau)``) define the SAME
model and share one run -- and picks the tau with the highest log evidence. All taus are
stored with their generations, rounding error and the evidence of their model.

Usage (overrides after ``--`` select the dataset: preset, scenario, design, seed)::

    uv run python scripts/x3b_tau_select.py --out results/x3b/gw_tau/S1_ref/real_like/seed0 \\
        --taus 18:32:1 --jobs 2 -- +experiment=x3b_gw_exact scenario=S1_ref seed=0

Writes ``<out>/tau_<tau>/`` (one run per distinct model; the representative tau is the one
with the smallest rounding error), ``<out>/tau_selection.csv`` (every tau) and
``<out>/best.json``. ``--collect`` only rebuilds the tables from finished runs. Ties
(identical generations) are reported, not broken: the evidence cannot distinguish them.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd


def parse_taus(spec: str) -> list[float]:
    """``a:b:step`` (inclusive) or a comma list."""
    if ":" in spec:
        a, b, step = (float(x) for x in spec.split(":"))
        n = int(math.floor((b - a) / step + 1e-9)) + 1
        return [round(a + i * step, 6) for i in range(n)]
    return [float(x) for x in spec.split(",")]


def generations(horizons: list[float], tau: float) -> tuple[int, ...]:
    """Generations simulated for each horizon (as ``GaltonWatson.generations``)."""
    return tuple(int(math.floor(h / tau + 0.5)) for h in horizons)


def rounding_error(horizons: list[float], tau: float) -> float:
    """Largest ``|h / tau - g|`` over the horizons."""
    return max(abs(h / tau - g) for h, g in zip(horizons, generations(horizons, tau), strict=True))


def horizons_of(overrides: list[str]) -> list[float]:
    """Design horizons (hours) of the composed config."""
    from hydra import compose, initialize_config_module

    with initialize_config_module("bio_sbi.conf", version_base="1.3"):
        cfg = compose("config", overrides=list(overrides))
    return [float(h) for h in cfg.design.horizons]


def plan(taus: list[float], horizons: list[float]) -> pd.DataFrame:
    """One row per tau: generations, rounding error, representative tau of its model."""
    rows = [{"tau": t, "generations": generations(horizons, t),
             "rounding_error": rounding_error(horizons, t)} for t in taus]  # fmt: skip
    df = pd.DataFrame(rows)
    rep = {}
    for g, grp in df.groupby("generations", sort=False):
        rep[g] = float(grp.sort_values(["rounding_error", "tau"]).tau.iloc[0])
    df["representative_tau"] = df.generations.map(rep)
    return df


def run_dir(out: Path, tau: float) -> Path:
    return out / f"tau_{tau:g}"


def launch(out: Path, tau: float, overrides: list[str], command: list[str]) -> int:
    d = run_dir(out, tau)
    if (d / "metrics.json").exists():
        return 0
    d.mkdir(parents=True, exist_ok=True)
    cmd = [*command, *overrides, f"hours_per_generation={tau:g}", f"hydra.run.dir={d}"]
    with open(d.parent / f"tau_{tau:g}.log", "w") as log:
        return subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)


def collect(out: Path, df: pd.DataFrame) -> pd.DataFrame:
    """Add each model's evidence (from its representative run) to every tau row."""
    ev = {}
    for tau in df.representative_tau.unique():
        f = run_dir(out, tau) / "metrics.json"
        if f.exists():
            m = json.loads(f.read_text())
            ho = m.get("heldout") or {}
            ev[tau] = {"log_evidence": m.get("log_evidence"),
                       "log_evidence_se": m.get("log_evidence_se"),
                       "ppc_out95": (m.get("ppc") or {}).get("fraction_outside_95"),
                       "heldout_lpd": ho.get("lpd"),
                       "max_rhat": (m.get("exact_gw") or {}).get("max_rhat"),
                       "run": str(run_dir(out, tau))}  # fmt: skip
    df = df.copy()
    for key in ("log_evidence", "log_evidence_se", "ppc_out95", "heldout_lpd", "max_rhat", "run"):
        df[key] = df.representative_tau.map(lambda t, k=key: ev.get(t, {}).get(k))
    df["generations"] = df.generations.map(lambda g: "/".join(map(str, g)))
    return df


def best(df: pd.DataFrame) -> dict:
    ok = df.dropna(subset=["log_evidence"])
    if ok.empty:
        return {}
    top = ok.log_evidence.max()
    win = ok[ok.log_evidence == top]
    second = ok[ok.generations != win.generations.iloc[0]].log_evidence.max()
    rep = win.sort_values(["rounding_error", "tau"]).iloc[0]
    return {"tau": float(rep.tau), "taus_same_model": win.tau.tolist(),
            "generations": rep.generations, "log_evidence": float(top),
            "log_evidence_se": rep.log_evidence_se,
            "log_bayes_factor_vs_next_model": float(top - second) if pd.notna(second) else None,
            "run": rep.run}  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    overrides = []
    if "--" in argv:
        i = argv.index("--")
        argv, overrides = argv[:i], argv[i + 1 :]
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--taus", default="18:32:1", help="a:b:step (inclusive) or comma list (h)")
    p.add_argument("--jobs", type=int, default=1, help="runs in parallel (each uses its threads)")
    p.add_argument("--collect", action="store_true", help="only collect finished runs")
    p.add_argument("--command", default="bio-sbi", help="run command (default: bio-sbi)")
    args = p.parse_args(argv)
    if not any(o.startswith("+experiment=") for o in overrides):
        p.error("give the dataset overrides after --, e.g. +experiment=x3b_gw_exact scenario=S1_ref seed=0")
    args.out.mkdir(parents=True, exist_ok=True)
    hours = horizons_of(overrides)
    df = plan(parse_taus(args.taus), hours)
    reps = sorted(df.representative_tau.unique())
    print(f"{len(df)} taus -> {len(reps)} distinct models: "
          + ", ".join(f"{t:g} h ({'/'.join(map(str, generations(hours, t)))})" for t in reps))
    if not args.collect:
        with ThreadPoolExecutor(max(args.jobs, 1)) as pool:
            codes = list(pool.map(lambda t: launch(args.out, t, overrides, args.command.split()),
                                  reps))  # fmt: skip
        failed = [t for t, c in zip(reps, codes, strict=True) if c]
        if failed:
            print(f"failed runs (see logs): {failed}")
    table = collect(args.out, df)
    table.to_csv(args.out / "tau_selection.csv", index=False)
    b = best(table)
    (args.out / "best.json").write_text(json.dumps({**b, "overrides": overrides}, indent=2,
                                                   default=float))  # fmt: skip
    pd.set_option("display.width", 200)
    print(table.drop(columns=["run"]).to_string(index=False))
    print("best:", b)


if __name__ == "__main__":
    main()
