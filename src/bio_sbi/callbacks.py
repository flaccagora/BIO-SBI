"""Hydra callbacks.

:class:`ExperimentInfo` writes ``info.md`` into every multirun (sweep) directory so a
results folder documents itself: what question the sweep answers (from the ``info``
block of an experiment preset), the exact command-line overrides, the swept grid, the
fixed settings that matter, the code version, and a standard **Data** section (synthetic
data, data model, true parameters, design, data seeds; see :mod:`bio_sbi.datacard`)
that is filled in from the runs when the multirun ends. Results are appended by the
user or by analysis scripts afterwards.

Registered in ``conf/config.yaml`` under ``hydra.callbacks``.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

from hydra.experimental.callback import Callback
from omegaconf import DictConfig, OmegaConf

from .datacard import SYNTHETIC, TITLE, write_data_section
from .provenance import _git


def _fmt(value: Any) -> str:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=False)
    return str(value)


def render_info(config: DictConfig) -> str:
    """Build the Markdown text of ``info.md`` for a sweep config."""
    info = config.get("info") or {}
    hydra = config.hydra
    title = info.get("title") or "Multirun"
    commit = _git("rev-parse", "--short", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    overrides = list(hydra.overrides.task)
    params = hydra.sweeper.get("params") or {}
    lines = [
        f"# {title}",
        "",
        f"**Launched:** {_dt.datetime.now():%Y-%m-%d %H:%M} · **Code:** commit `{commit}`"
        + (" (with uncommitted changes)" if dirty else ""),
        "",
    ]
    if info.get("question"):
        lines += ["## Question", "", str(info.question).strip(), ""]
    lines += ["## Command", "", "```bash", "uv run bio-sbi -m " + " ".join(overrides), "```", ""]
    lines += ["## Grid", ""]
    # Swept values come from the preset's sweeper params and from comma lists on the CLI.
    # A CLI value for the same key replaces the preset's list (even a single value).
    swept = {str(k): _fmt(v) for k, v in dict(params).items()}
    for item in overrides:
        key, _, value = item.partition("=")
        key = key.lstrip("+")
        if "," in value and not value.startswith(("[", "{")):
            swept[key] = value
        elif key in swept:
            del swept[key]  # pinned to one value on the command line
    if swept:
        lines += ["| Parameter | Values |", "|---|---|"]
        lines += [f"| `{k}` | {_fmt(v)} |" for k, v in swept.items()]
    else:
        lines += ["No swept parameters."]

    def show(key: str, value: str) -> str:
        return "swept (see Grid)" if key in swept else value

    data_overrides = [o for o in overrides if "data." in o.split("=", 1)[0]]
    data = (
        f"`simulator={config.data.simulator.name}`, `law={config.data.law.name}` "
        f"(overrides: {', '.join(f'`{o}`' for o in data_overrides)})"
        if data_overrides
        else "same as the inference model"
    )
    design_name = config.design.get("name", "custom")
    lines += [
        "",
        "## Fixed settings",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| simulator | {show('simulator', f'`{config.simulator.name}`')} |",
        f"| law (inference model) | {show('law', f'`{config.law.name}`')} |",
        f"| data model | {data} |",
        f"| inference | {show('inference', f'`{config.inference.name}`')}, "
        f"rounds={config.inference.method.rounds} |",
        f"| design | {show('design', f'`{design_name}`')} |",
        f"| seed | {show('seed', str(config.seed))} |",
        f"| budget | {_fmt(config.budget)} |",
        "",
    ]
    if info.get("notes"):
        lines += ["## Notes", "", str(info.notes).strip(), ""]
    lines += [
        TITLE,
        "",
        SYNTHETIC,
        "",
        "_Completed automatically from the runs (`.hydra/config.yaml`, `reference.npz`) when",
        "the multirun ends; refresh with `uv run python scripts/add_data_section.py <dir>`._",
        "",
        "## Results",
        "",
        "One directory per job (`metrics.json`, `recovery.csv`, posterior samples, SBC, PPC,",
        "figures, `.hydra/`). Aggregate with `uv run bio-sbi-collect <this directory>`.",
        "",
    ]
    return "\n".join(lines)


class ExperimentInfo(Callback):
    """Write ``<sweep dir>/info.md`` at multirun start; fill its Data section at the end."""

    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None:
        """Create the sweep directory and its ``info.md``."""
        sweep_dir = Path(config.hydra.sweep.dir)
        sweep_dir.mkdir(parents=True, exist_ok=True)
        path = sweep_dir / "info.md"
        if not path.exists():
            path.write_text(render_info(config))

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        """Insert or refresh the Data section from the finished runs."""
        sweep_dir = Path(config.hydra.sweep.dir)
        if sweep_dir.exists():
            write_data_section(sweep_dir)
