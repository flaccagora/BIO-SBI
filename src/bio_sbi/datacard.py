"""The standard "Data" section of a results folder's ``info.md``.

Every experiment documents the data it was run on: they are synthetic, which data model
generated them (simulator, law, observation, founders), the full true parameters (or
that they were drawn from the prior), the design, and the data seeds / number of
distinct reference datasets. :func:`describe_data` builds that section from the runs
themselves (each job's ``.hydra/config.yaml`` and ``reference.npz``), so it is correct
even when the data model is swept or overridden on the command line.

Used by the Hydra callback (:mod:`bio_sbi.callbacks`) at the end of every multirun and
by ``scripts/add_data_section.py`` for existing folders.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from .simulators.law import flatten

TITLE = "## Data"
SYNTHETIC = (
    '**Synthetic data.** Every reference ("observed") dataset below was simulated by the '
    "data model described here; no measured data were used."
)


def _value(cfg, key: str, default=None):
    try:
        value = OmegaConf.select(cfg, key, default=default, throw_on_resolution_failure=False)
    except Exception:  # noqa: BLE001 - a broken interpolation must not break documentation
        return default
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    return value


def _fmt(value) -> str:
    if value is None:
        return "default"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {_fmt(v)}" for k, v in value.items()) + "}"
    return str(value)


def job_record(job: Path) -> dict:
    """Data-side description of one run (from its saved config and reference data)."""
    cfg = OmegaConf.load(job / ".hydra" / "config.yaml")
    ref = np.load(job / "reference.npz")
    horizons = sorted(
        (float(k.removeprefix("counts_h")), ref[k].shape[0]) for k in ref.files
        if k.startswith("counts_h")
    )  # fmt: skip
    digest = hashlib.sha1()
    for k in sorted(k for k in ref.files if k.startswith("counts_h")):
        digest.update(np.ascontiguousarray(ref[k]).tobytes())
    names = [str(n) for n in ref["truth_names"]] if "truth_names" in ref.files else None
    truth = dict(zip(names, ref["truth"].tolist(), strict=True)) if names else {}
    model = cfg.data.simulator if "data" in cfg else cfg.simulator
    law = cfg.data.law if "data" in cfg else cfg.law
    observation = cfg.data.observation if "data" in cfg else cfg.observation
    data_seed = _value(cfg, "data_seed", None)
    fields = {
        "simulator": _value(model, "name"),
        "time unit": _value(model, "model.time_unit", "generations"),
        "lifetime shape": _value(model, "model.lifetime_shape"),
        "law": _value(law, "name"),
        "types": _value(law, "types"),
        "founder sizes": _value(law, "founder_sizes", [1]),
        "fixed in data law": _fmt(flatten(_value(law, "fixed", {}) or {})) or "none",
        "measured labels": _value(observation, "labels"),
        "capture": _value(observation, "capture"),
        "confusion": _value(observation, "confusion"),
        "selection": _value(observation, "selection"),
        "truth source": _value(cfg, "truth.source", "reference"),
        "horizons": [h for h, _ in horizons],
        "clones per horizon": [n for _, n in horizons],
    }
    # Spell out what a null setting means (see observation.py / simulators).
    defaults = {
        "measured labels": "same as types",
        "capture": "1 for every type",
        "confusion": "none (identity)",
        "lifetime shape": "n/a (discrete generations)",
    }
    return {
        "job": job,
        "fields": {
            k: defaults.get(k, "default") if v is None else _fmt(v) for k, v in fields.items()
        },  # fmt: skip
        "truth": truth,
        "data_seed": int(data_seed if data_seed is not None else _value(cfg, "seed", 0)),
        "dataset": digest.hexdigest()[:12],
    }


def _truth_lines(truth: dict) -> list[str]:
    """Compact rows: one per simplex row / scalar prefix, e.g. ``A: death 0.03, ...``."""
    rows: dict[str, list[str]] = defaultdict(list)
    for name, value in truth.items():
        prefix, _, entry = name.rpartition(".")
        label = f"{entry} cell(s): {value:g}" if prefix == "founder_size" else f"{entry} {value:g}"
        rows[prefix or name].append(label)
    return [f"| {prefix} | {', '.join(items)} |" for prefix, items in rows.items()]


def describe_data(root: str | Path) -> str:
    """Markdown "Data" section for all runs (``reference.npz``) below ``root``."""
    root = Path(root)
    jobs = sorted(p.parent for p in root.rglob("reference.npz") if (p.parent / ".hydra").exists())
    if not jobs:
        return f"{TITLE}\n\n{SYNTHETIC}\n\n_No finished runs with reference data yet._\n"
    records = [job_record(job) for job in jobs]
    keys = list(records[0]["fields"])
    varying = [k for k in keys if len({r["fields"][k] for r in records}) > 1]
    constant = [k for k in keys if k not in varying]
    names = list(dict.fromkeys(n for r in records for n in r["truth"]))
    truth_vary = [
        n for n in names if len({round(r["truth"].get(n, np.nan), 12) for r in records}) > 1
    ]
    truth_const = {
        n: records[0]["truth"][n] for n in names if n not in truth_vary and n in records[0]["truth"]
    }
    seeds = sorted({r["data_seed"] for r in records})
    datasets = {r["dataset"] for r in records}
    prior = any("prior" in r["fields"]["truth source"] for r in records)

    lines = [
        TITLE,
        "",
        SYNTHETIC,
        "",
        f"{len(records)} runs below this folder · **{len(datasets)} distinct reference "
        f"datasets** · data seeds: {_fmt(seeds)}. Runs with the same data model, truth, "
        "design and data seed share the same reference dataset.",
        "",
        "### Data model",
        "",
        "| Setting | Value |",
        "|---|---|",
        *[f"| {k} | {records[0]['fields'][k]} |" for k in constant],
    ]
    if varying:
        lines.append(f"| varies across runs | {', '.join(varying)} (see table below) |")
    lines += ["", "### True parameters", ""]
    if prior:
        lines += [
            "Some or all truths were **drawn from the data model's prior** (seeded by the "
            "data seed); the values actually used are listed per dataset.",
            "",
        ]
    if truth_const:
        lines += [
            "Common to all runs"
            + (" (other parameters vary, see below)" if truth_vary else "")
            + ":",
            "",
            "| Row | Values |",
            "|---|---|",
            *_truth_lines(truth_const),
            "",
        ]
    if varying or truth_vary:
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for r in records:
            key = tuple(r["fields"][k] for k in varying) + tuple(
                f"{r['truth'].get(n, float('nan')):g}" for n in truth_vary
            )
            groups[key].append(r)
        header = [*varying, *truth_vary, "runs", "datasets", "data seeds"]
        lines += [
            "Per data setting:",
            "",
            "| " + " | ".join(header) + " |",
            "|" + "---|" * len(header),
        ]
        for key, members in sorted(groups.items()):
            cells = [
                *key,
                str(len(members)),
                str(len({m["dataset"] for m in members})),
                _fmt(sorted({m["data_seed"] for m in members})),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "Source: each run's `.hydra/config.yaml` (data side: `data.simulator`, `data.law`, "
        "`data.observation`, `truth`, `data_seed`) and `reference.npz` (`truth`, "
        "`truth_names`, `counts_h<horizon>`). Generated by `bio_sbi.datacard`.",
        "",
    ]
    return "\n".join(lines)


def upsert_section(text: str, section: str, title: str = TITLE) -> str:
    """Replace the ``title`` section of a Markdown text, or insert it before Results."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == title), None)
    if start is not None:
        end = next(
            (j for j in range(start + 1, len(lines)) if lines[j].startswith("## ")), len(lines)
        )
        return "\n".join([*lines[:start], *section.rstrip("\n").splitlines(), "", *lines[end:]])
    anchor = next(
        (i for i, line in enumerate(lines) if line.startswith(("## Results", "## Reading"))),
        None,
    )
    if anchor is None:
        return text.rstrip("\n") + "\n\n" + section
    return "\n".join([*lines[:anchor], *section.rstrip("\n").splitlines(), "", *lines[anchor:]])


def write_data_section(root: str | Path) -> Path:
    """Insert or refresh the Data section of ``root/info.md`` (created if missing)."""
    root = Path(root)
    path = root / "info.md"
    text = path.read_text() if path.exists() else f"# {root.name}\n"
    path.write_text(upsert_section(text, describe_data(root)))
    return path
