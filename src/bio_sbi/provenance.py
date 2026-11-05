"""Run provenance: software versions, git state and hardware, saved with every run.

Hydra already stores the fully composed config in ``<run>/.hydra/config.yaml`` and the
command-line overrides in ``<run>/.hydra/overrides.yaml``; :func:`write_manifest`
complements this with everything needed to reproduce the software environment.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

PACKAGES = ("bio-sbi", "sbi", "torch", "numpy", "scipy", "numba", "hydra-core")


def _git(*args: str) -> str | None:
    root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _git_state() -> tuple[str | None, bool | None]:
    status = _git("status", "--porcelain")
    return _git("rev-parse", "HEAD"), (bool(status) if status is not None else None)


#: Git state when this module was imported, i.e. the code a (multirun) process executes.
#: A multirun keeps its imported code even if commits happen while it runs, so the HEAD
#: at run time can differ from the code that ran.
_LOADED_COMMIT, _LOADED_DIRTY = _git_state()


def manifest() -> dict:
    """Collect environment metadata as a JSON-serialisable dict.

    ``git_commit``/``git_dirty`` describe the code loaded by this process;
    ``git_commit_at_run`` is the repository HEAD when the run started (they differ if
    commits were made while a long multirun was executing).
    """
    import torch

    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    head, _ = _git_state()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "git_commit": _LOADED_COMMIT,
        "git_dirty": _LOADED_DIRTY,
        "git_commit_at_run": head,
        "cuda": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
    }


def write_manifest(directory: str | Path) -> None:
    """Write :func:`manifest` to ``directory/manifest.json``."""
    Path(directory, "manifest.json").write_text(json.dumps(manifest(), indent=2))
