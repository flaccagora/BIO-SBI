"""Config files: three_split, three_type_traj, real_like design and experiment presets."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_module
from hydra.core.override_parser.overrides_parser import OverridesParser

from bio_sbi.callbacks import render_info
from bio_sbi.pipeline import build_components, build_data_generator, reference_parameters

PRESETS = sorted(
    p.stem for p in (Path(__file__).parents[1] / "src/bio_sbi/conf/experiment").glob("*.yaml")
)


def compose_cfg(*overrides, hydra=False):
    with initialize_config_module("bio_sbi.conf", version_base="1.3"):
        return compose("config", overrides=list(overrides), return_hydra_config=hydra)


def build(cfg):
    comp = build_components(cfg)
    data = build_data_generator(cfg, comp.generator)
    return comp, data, reference_parameters(cfg, data.simulator)


def test_three_split_q_and_three_type_traj():
    comp, data, _ = build(
        compose_cfg(
            "law=three_type_traj",
            "observation=three_split",
            "observation.q=0.25",
            "law@data.law=three_type",
            "observation@data.observation=three_split",
            "data.observation.q=1",
        )
    )
    assert comp.generator.observation.confusion[2].tolist() == [0.25, 0.75]
    assert data.observation.confusion[2].tolist() == [1.0, 0.0]
    free = set(comp.simulator.space.free_names)
    assert not free & {"C.to_A", "C.div_AA", "C.div_AB", "C.div_AC"}
    assert {"C.to_B", "C.div_BC", "A.to_C", "B.to_C"} <= free


@pytest.mark.parametrize(
    ("simulator", "horizons"),
    [
        ("galton_watson", (6.0, 10.0)),
        ("galton_watson_hours", (144.0, 240.0)),
        ("bellman_harris", (144.0, 240.0)),
    ],
)
def test_real_like_design_follows_simulator(simulator, horizons):
    comp, _, _ = build(compose_cfg(f"simulator={simulator}", "design=real_like"))
    assert comp.generator.design.cohorts == tuple(zip(horizons, (165, 101), strict=True))


def sweep_points(name):
    """Overrides varying one swept key at a time (seed excluded) for a preset."""
    hcfg = compose_cfg(f"+experiment={name}", hydra=True)
    parser = OverridesParser.create()
    points = [[]]
    for key, value in (hcfg.hydra.sweeper.get("params") or {}).items():
        if key != "seed":
            (override,) = parser.parse_overrides([f"{key}={value}"])
            points += [[f"{key}={v}"] for v in override.sweep_string_iterator()]
    return hcfg, points


@pytest.mark.parametrize("name", PRESETS)
def test_experiment_presets_compose_and_build(name):
    hcfg, points = sweep_points(name)
    if name.startswith(("e1", "e2", "e3", "e4", "x2", "x3")):
        assert hcfg.info.title and hcfg.info.question
        assert hcfg.info.title in render_info(hcfg)
    for point in points:
        _, data, truth = build(compose_cfg(f"+experiment={name}", *point))
        assert len(truth) == len(data.simulator.parameter_names)


def test_preset_overrides_reach_the_intended_model():
    def truth_of(*overrides):
        comp, data, truth = build(compose_cfg(*overrides))
        return comp, data, dict(zip(data.simulator.parameter_names, truth, strict=True))

    comp, _, truth = truth_of("+experiment=e1c_fixed_death", "law.fixed.A.death=0.2")
    assert (
        truth["A.death"] == pytest.approx(0.03) and "A.death" not in comp.simulator.space.free_names
    )
    comp, data, _ = truth_of("+experiment=e2b_typing_ignored", "data.observation.confusion=0.1")
    assert (
        comp.generator.observation.confusion[0, 1] == 0 and data.observation.confusion[0, 1] == 0.1
    )
    comp, data, _ = truth_of("+experiment=e2a_typing_known", "observation.confusion=0.1")
    assert comp.generator.observation.confusion[0, 1] == data.observation.confusion[0, 1] == 0.1
    comp, data, truth = truth_of(
        "+experiment=e3a_lifetime_fixed", "lifetime_setting={B:45,T:24.375}"
    )
    assert truth["B.mean_lifetime"] == 45 and comp.simulator.space.fixed_scalars == {
        "A.mean_lifetime": 24.375,
        "B.mean_lifetime": 24.375,
    }
    _, _, truth = truth_of(
        "+experiment=e1a_founder_number", "data.law.reference.founder_size={1:0.6,2:0.3,3:0.1}"
    )
    assert truth["founder_size.1"] == pytest.approx(0.6) and truth["founder.A"] == 1.0
