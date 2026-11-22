"""How often does the XG2 MLE sit on a face of the simplex, and along which outcomes?

For every XG2 run (n = 128 ... 4096, seeds 0-2): the outcomes with MLE probability below
1e-3 whose truth is not 0, and the share of that outcome's unit vector (projected on the
tangent space) inside the weak-4 Fisher subspace of the truth (random baseline 0.4).
Writes ``tables/mle_faces.{csv,md}``.

  uv run python scripts/weak_directions/mle_faces.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import NAMES, OUT, TAB, TRUTHS, XG2, row_contrasts  # noqa: E402

TOL = 1e-3


def main() -> None:
    z = np.load(OUT / "fisher_eigen.npz")
    U = row_contrasts()
    rows = []
    for gt in TRUTHS:
        W = z[f"{gt}_V"][:, :4]
        for f in sorted((XG2 / gt).glob("n*_s*/mle.json")):
            d = json.loads(f.read_text())
            n, s = (int(x) for x in f.parent.name[1:].split("_s"))
            mle = np.array([d["mle"][k] for k in NAMES])
            tr = np.array([d["truth"][k] for k in NAMES])
            hit = [i for i in range(12) if mle[i] < TOL and tr[i] > 0]
            share = []
            for i in hit:
                e = U @ (U.T @ np.eye(12)[i])
                e /= np.linalg.norm(e)
                share.append(float(np.sum((W.T @ e) ** 2)))
            rows.append({"truth": gt, "n": n, "seed": s, "n_face_outcomes": len(hit),
                         "face_outcomes": " ".join(NAMES[i] for i in hit),
                         "weak4_share_of_face_outcomes": float(np.mean(share)) if share
                         else np.nan})  # fmt: skip
    order = {g: i for i, g in enumerate(TRUTHS)}
    df = pd.DataFrame(rows)
    df = df.assign(o=df.truth.map(order)).sort_values(["o", "n", "seed"]).drop(columns="o")
    df.to_csv(TAB / "mle_faces.csv", index=False)
    summ = df.assign(on_face=df.n_face_outcomes > 0).groupby("n").agg(
        runs=("on_face", "size"), mle_on_interior_face=("on_face", "sum"),
        weak4_share=("weak4_share_of_face_outcomes", "mean")).reset_index()  # fmt: skip
    counts = df.face_outcomes.str.split().explode().value_counts()
    (TAB / "mle_faces.md").write_text(
        "MLE outcomes < 1e-3 whose truth is > 0 (a face the truth is not on)\n\n"
        + summ.to_markdown(index=False, floatfmt=".2f") + "\n\nOutcomes hit:\n\n"
        + counts.to_frame().to_markdown() + "\n\n"
        + df.to_markdown(index=False, floatfmt=".2f") + "\n")
    print(summ)
    print(counts)


if __name__ == "__main__":
    main()
