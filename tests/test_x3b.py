"""X3b tooling: tau selection by evidence (driver), presets, analysis helpers."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import x3b_tau_select as ts  # noqa: E402


def test_tau_grid_is_deduplicated_by_generations():
    df = ts.plan(ts.parse_taus("18:32:1"), [144.0, 240.0])
    groups = df.groupby("representative_tau").tau.apply(list).to_dict()
    assert groups[24.0] == [23.0, 24.0, 25.0]  # 6/10 generations; 24 h rounds exactly
    assert groups[30.0] == [29.0, 30.0, 31.0, 32.0]
    assert len(groups) == 7
    assert ts.generations([144.0, 240.0], 25.0) == (6, 10)
    assert ts.rounding_error([144.0, 240.0], 25.0) == pytest.approx(0.4)
    assert ts.parse_taus("20,24") == [20.0, 24.0]


def test_collect_and_best_pick_the_highest_evidence(tmp_path):
    df = ts.plan([20.0, 24.0, 25.0, 30.0], [144.0, 240.0])
    for tau, z in ((20.0, -1710.0), (24.0, -1700.0), (30.0, -1800.0)):
        d = ts.run_dir(tmp_path, tau)
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps(
            {"log_evidence": z, "log_evidence_se": 0.05, "ppc": {"fraction_outside_95": 0.0},
             "exact_gw": {"max_rhat": 1.0}}))  # fmt: skip
    table = ts.collect(tmp_path, df)
    assert table.set_index("tau").log_evidence[25.0] == -1700.0  # same model as 24 h
    b = ts.best(table)
    assert b["tau"] == 24.0 and b["taus_same_model"] == [24.0, 25.0]
    assert b["log_bayes_factor_vs_next_model"] == pytest.approx(10.0)
    assert b["generations"] == "6/10"


@pytest.mark.slow
def test_tau_select_end_to_end(tmp_path):
    over = ["+experiment=x3b_gw_exact", "scenario=S3_equal_var", "seed=0",
            "design.clones=8", "design.horizons=[48,72]", "budget.ppc_samples=0",
            "inference.method.chains=2", "inference.method.warmup=30",
            "inference.method.draws=40", "inference.method.init_starts=2",
            "inference.method.threads=2", "inference.method.evidence_draws=100",
            "workers=1"]  # fmt: skip
    ts.main(["--out", str(tmp_path), "--taus", "22,24,26", "--jobs", "2", "--", *over])
    table = pd.read_csv(tmp_path / "tau_selection.csv")
    assert table.log_evidence.notna().all()
    b = json.loads((tmp_path / "best.json").read_text())
    assert b["tau"] in (22.0, 24.0, 26.0)
    # 22 h and 24 h: 2/3 generations (48 h, 72 h) -> one shared run
    assert table.set_index("tau").representative_tau[22.0] == table.set_index(
        "tau").representative_tau[24.0]  # fmt: skip
    assert len(list(tmp_path.glob("tau_*/metrics.json"))) == 2


@pytest.mark.parametrize("preset", ["x3b_bh_exact", "x3b_gw_exact"])
@pytest.mark.parametrize("scenario", ["S1_ref", "S2_equal_sync", "S3_equal_var", "S4_desync"])
def test_presets_share_the_data_model(preset, scenario):
    from hydra import compose, initialize_config_module

    with initialize_config_module("bio_sbi.conf", version_base="1.3"):
        cfg = compose("config", overrides=[f"+experiment={preset}", f"scenario={scenario}"])
    spec = {"S1_ref": (25, 30, 8), "S2_equal_sync": (25, 25, 32), "S3_equal_var": (25, 25, 4),
            "S4_desync": (25, 30, 2)}[scenario]  # fmt: skip
    d = cfg.data.simulator
    assert d.name == "bellman_harris"
    assert (d.reference.A.mean_lifetime, d.reference.B.mean_lifetime,
            d.model.lifetime_shape) == spec  # fmt: skip
    if preset == "x3b_bh_exact":
        assert cfg.simulator.model.lifetime_shape == spec[2]
        assert cfg.inference.method.reparam == "stable+probit"
        assert cfg.inference.method.dt * spec[2] <= 32  # DOP853 stability (shape 32: dt 1)
    else:
        assert cfg.simulator.model.hours_per_generation == 24
