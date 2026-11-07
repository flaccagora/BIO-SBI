"""scripts/prepare_real_data.py on tiny synthetic organoid tables."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_real_data.py"
COLUMNS = ["organoid", "n_KRT20", "n_LGR5", "n_cells",
           "n_cells_3D_sample1", "n_cells_3D_sample2", "n_cells_3D_sample3",
           "LGR5_sample1", "LGR5_sample2", "LGR5_sample3",
           "KRT20_sample1", "KRT20_sample2", "KRT20_sample3"]  # fmt: skip


@pytest.fixture(scope="module")
def prep():
    spec = importlib.util.spec_from_file_location("prepare_real_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def data_dir(tmp_path):
    day6 = pd.DataFrame(
        [
            ["o6_1", 5, 20, 25, 155, 126, 116, 124, 96, 99, 31, 30, 17],
            ["o6_2", 3, 3, 6, 14, 0, 22, 7, 0, 10, 7, 0, 12],  # failed 3D sample 2
        ],
        columns=COLUMNS,
    )
    day10 = pd.DataFrame(
        [["o10_1", 24, 23, 50, 279, 286, 253, 137, 155, 122, 142, 131, 121]], columns=COLUMNS
    )  # 3 cells of neither type in 2D, 10 in 3D sample 3
    folder = tmp_path / "raw"
    folder.mkdir()
    day6.to_csv(folder / "df_6D_withmapping.csv", sep="\t", index=False)
    day10.insert(0, "Unnamed: 0", [0])  # like the real file: ",Unnamed: 0,organoid,..."
    day10.to_csv(folder / "df_10D_withmapping_new.csv")
    return folder


def test_read_table_detects_delimiter(prep, data_dir):
    six = prep.read_table(data_dir / "df_6D_withmapping.csv")
    ten = prep.read_table(data_dir / "df_10D_withmapping_new.csv")
    assert list(six.columns) == COLUMNS and list(ten.columns) == COLUMNS
    assert ten["n_LGR5"].iloc[0] == 23


def test_convert_writes_all_variants(prep, data_dir, tmp_path):
    out = tmp_path / "real"
    prep.main(["--data-dir", str(data_dir), "--out", str(out)])
    names = {p.name for p in out.glob("*.csv")}
    variants = ["3d_mean", "3d_sample1", "3d_sample2", "3d_sample3", "2d"]
    assert names == {f"crc0322_{v}_{u}.csv" for v in variants for u in ("bh", "gw")}

    mean = pd.read_csv(out / "crc0322_3d_mean_bh.csv")
    assert list(mean.columns) == ["horizon", "A", "B"]
    assert mean["horizon"].tolist() == [144, 144, 240]
    # round(mean of three samples), zeros included: (124+96+99)/3 = 106.33 -> 106
    assert mean.iloc[0].tolist() == [144, 106, 26]
    assert mean.iloc[1].tolist() == [144, 6, 6]  # (7+0+10)/3 = 5.67, (7+0+12)/3 = 6.33
    assert pd.read_csv(out / "crc0322_3d_mean_gw.csv")["horizon"].tolist() == [6, 6, 10]

    two_d = pd.read_csv(out / "crc0322_2d_gw.csv")
    assert two_d.values.tolist() == [[6, 20, 5], [6, 3, 3], [10, 23, 24]]
    sample2 = pd.read_csv(out / "crc0322_3d_sample2_bh.csv")
    assert len(sample2) == 2  # the 0-cell estimate is dropped
    assert (sample2[["A", "B"]].sum(axis=1) > 0).all()

    text = (out / "README.md").read_text()
    assert "o6_2" in text and "3D sample 2 is 0 cells" in text
    assert "| 10 | 2d | 1 | 0.060 |" in text  # 3 of 50 cells are neither type


def test_quality_flags(prep, data_dir):
    table = prep.read_table(data_dir / "df_6D_withmapping.csv")
    assert prep.quality_flags(table, 6) == [
        "day 6: 3D sample 2 is 0 cells for 1 organoid(s) whose 2D section is non-empty (o6_2)"
    ]
    table["n_LGR5"] = table["n_LGR5"].astype(float)
    table["n_cells"] = table["n_cells"].astype(float)
    table.loc[0, "n_LGR5"] = None
    table.loc[1, "n_cells"] = 2.5
    flags = " ".join(prep.quality_flags(table, 6))
    assert "missing values" in flags and "non-integer" in flags
