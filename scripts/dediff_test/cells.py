"""Simulation cells of the dedifferentiation-test study (truths, typing, designs)."""

from __future__ import annotations

import numpy as np

from bio_sbi.inference import dediff_lrt as D

HORIZONS = (4, 6, 8, 10)
NS = (128, 256, 512, 1024)
N_MAX = max(NS)

#: Reference A row (all null and alternative truths).
A_ROW = {"death": 0.03, "persist": 0.03, "to_B": 0.05, "div_AA": 0.55, "div_AB": 0.05,
         "div_BB": 0.29}  # fmt: skip

#: Null B rows (no dedifferentiation).
NULLS = {
    "N1": {"death": 0.2, "persist": 0.8},                    # B persists (div_BB = 0: boundary)
    "N2": {"death": 0.1, "persist": 0.5, "div_BB": 0.4},     # B self-renews
    "N3": {"death": 0.5, "persist": 0.5},                    # fast B loss (div_BB = 0)
    "N4": {"death": 0.2, "persist": 0.3, "div_BB": 0.5},     # B self-renews, more turnover
}
#: Alternatives: base null, route, effect size m_BA = to_A + 2 div_AA + div_AB.
ALT_BASES = ("N1", "N2")
ROUTES = {
    "toA": {"to_A": 1.0},                                    # switch only
    "divAB": {"div_AB": 1.0},                                # B -> A + B only
    "mixed": {"to_A": 0.5, "div_AB": 0.25, "div_AA": 0.125},  # m/2, m/4, m/8
}
# requested grid 0.02-0.4, extended downwards after the pilot (power ~1 already at 0.05)
M_VALUES = (0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4)
#: Mis-typing of the data (true type -> label).
TYPINGS = {
    "sym0.02": (0.02, "symmetric"), "sym0.05": (0.05, "symmetric"),
    "bA0.02": (0.02, "b_as_a"), "bA0.05": (0.05, "b_as_a"),
    # added after the pilot (2% already gives ~100% false positives when ignored)
    "bA0.005": (0.005, "b_as_a"), "bA0.01": (0.01, "b_as_a"),
}
TYPING_BASES = ("N1", "N2")


def alt_row(base: str, route: str, m: float) -> dict:
    """Base null B row with mass moved from persist to the route's entries (m_BA = m)."""
    row = dict(NULLS[base])
    moved = 0.0
    for e, w in ROUTES[route].items():
        row[e] = row.get(e, 0.0) + w * m
        moved += w * m
    row["persist"] -= moved
    if row["persist"] < 0:
        raise ValueError(f"{base} {route} m={m}: not enough persist mass")
    return row


def cell(name: str) -> dict:
    """Truth ``P``, data confusion and description of a cell name.

    ``N1`` (null), ``N1_toA_m0.05`` (alternative), ``N1_bA0.05`` (null data, mis-typed).
    """
    parts = name.split("_")
    base = parts[0]
    if base not in NULLS:
        raise ValueError(f"unknown base {base}")
    out = {"name": name, "base": base, "route": None, "m": 0.0, "typing": None,
           "confusion": None}  # fmt: skip
    B = dict(NULLS[base])
    if len(parts) == 3:
        route, m = parts[1], float(parts[2][1:])
        B = alt_row(base, route, m)
        out.update(route=route, m=m)
    elif len(parts) == 2:
        rate, kind = TYPINGS[parts[1]]
        out.update(typing=parts[1], confusion=D.confusion_matrix(rate, kind),
                   confusion_half=D.confusion_matrix(rate / 2, kind))  # fmt: skip
    out["P"] = D.P_from_rows(A_ROW, B)
    assert abs(D.m_ba(out["P"]) - out["m"]) < 1e-12
    return out


def study_cells(study: str) -> list[str]:
    """Cell names of a study."""
    if study == "null":
        return list(NULLS)
    if study == "alt":
        return [f"{b}_{r}_m{m:g}" for b in ALT_BASES for r in ROUTES for m in M_VALUES]
    if study == "typing":
        return [f"{b}_{t}" for b in TYPING_BASES for t in TYPINGS]
    raise ValueError(study)


def expected_cells(sim, P: np.ndarray, horizon: int = 10) -> float:
    """Expected cells per clone (unconditional) at ``horizon``."""
    return float(sim.expected_counts(D.physical(sim, P), horizon).sum())
