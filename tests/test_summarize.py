"""scripts/summarize.py on a synthetic results tree (no training)."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize.py"


@pytest.fixture(scope="module")
def summarize():
    spec = importlib.util.spec_from_file_location("summarize_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_run(path: Path, overrides: list[str], seed: int, error: float, covered: bool) -> None:
    (path / ".hydra").mkdir(parents=True)
    (path / ".hydra" / "overrides.yaml").write_text(yaml.safe_dump(overrides))
    params = [
        {"parameter": "A.death", "truth": 0.03, "posterior_mean": 0.03 + error,
         "posterior_std": 0.01, "abs_error": error, "contraction": 0.9, "covered": covered,
         "at_boundary": False, "inferred": True},
        {"parameter": "B.div_AA", "truth": 0.0, "posterior_mean": 0.0, "posterior_std": 0.0,
         "abs_error": 0.0, "contraction": float("nan"), "covered": False,
         "at_boundary": True, "inferred": False},
    ]  # fmt: skip
    metrics = {
        "simulator": "galton_watson", "law": "reversible_ab", "inference": "npe",
        "rounds": 1, "simulations": 100, "seed": seed, "data_seed": seed,
        "design": {"name": "real_like", "horizons": [6, 10], "clones": [165, 101]},
        "reference_detection": [
            {"horizon": 6.0, "seeded": None, "detection_rate": None, "detection_probability": 0.5},
        ],
        "training": [{"training_seconds": 1.0, "simulation_seconds": 1.0}],
        "total_seconds": 3.0,
        "recovery": {"mean_abs_error": error, "mean_contraction": 0.9, "coverage90": float(covered),
                     "per_parameter": params},
        "ppc": {"fraction_outside_95": 0.1, "mean_abs_z": 1.0},
        "calibration": {"cases": 10, "mean_abs_coverage_error": 0.02, "ks_pvalue": {"x": 0.3}},
    }  # fmt: skip
    (path / "metrics.json").write_text(json.dumps(metrics))


def test_summarize_groups_and_averages_over_seeds(tmp_path, summarize):
    root = tmp_path / "e2"
    for arm in ("known", "ignored"):
        for e in (0.0, 0.1):
            for seed in range(3):
                job = root / arm / f"{seed}_e={e},seed={seed}"
                ov = [f"data.observation.confusion={e}", f"seed={seed}", "simulator=galton_watson"]
                fake_run(job, ov, seed, error=e + 0.01 * seed, covered=seed < 2)
    groups, table = summarize.main([str(root)])
    assert list(groups.columns[:3]) == ["sweep", "data.observation.confusion", "runs"]
    assert len(groups) == 4 and set(groups["runs"]) == {3}
    assert groups["detection_probability_h6"].iloc[0] == 0.5
    assert set(table["parameter"]) == {"A.death"}  # only inferred parameters by default
    row = table[(table["sweep"] == "known") & (table["data.observation.confusion"] == "0.1")]
    assert row["abs_error"].item() == pytest.approx(0.11)
    assert row["coverage90"].item() == pytest.approx(2 / 3)
    assert not row["recovered"].item()
    assert (root / "summary_groups.csv").exists()
    assert (root / "summary_group_parameters.csv").exists()

    _, table = summarize.main([str(root), "--params", "B.*", "--group-by", "sweep"])
    assert set(table["parameter"]) == {"B.div_AA"} and table["coverage90"].isna().all()
    with pytest.raises(SystemExit):
        summarize.main([str(root), "--group-by", "nope"])
