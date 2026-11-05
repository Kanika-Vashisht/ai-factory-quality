from pathlib import Path

import pytest

from fqa.maintenance.cmapss import COLUMNS, load_subset


def test_columns_and_shapes(cmapss_root: Path) -> None:
    train, test = load_subset(cmapss_root, "FD001")
    assert list(train.columns) == [*COLUMNS, "rul"]
    assert len(COLUMNS) == 26
    assert len(train) == 8 and len(test) == 6


def test_train_rul_counts_down_to_zero(cmapss_root: Path) -> None:
    train, _ = load_subset(cmapss_root, "FD001")
    engine_1 = train[train["unit"] == 1]["rul"].tolist()
    assert engine_1 == [4, 3, 2, 1, 0]
    assert train.groupby("unit")["rul"].min().eq(0).all()


def test_test_rul_uses_rul_file(cmapss_root: Path) -> None:
    _, test = load_subset(cmapss_root, "FD001")
    # Unit 1: 2 cycles, RUL file says 10 at the last row -> [11, 10]
    assert test[test["unit"] == 1]["rul"].tolist() == [11, 10]
    # Unit 2: 4 cycles, RUL 7 at the last row -> [10, 9, 8, 7]
    assert test[test["unit"] == 2]["rul"].tolist() == [10, 9, 8, 7]


def test_rul_file_length_mismatch_raises(cmapss_root: Path) -> None:
    (cmapss_root / "RUL_FD001.txt").write_text("10 \n")
    with pytest.raises(ValueError, match="units"):
        load_subset(cmapss_root, "FD001")


def test_missing_subset_gives_helpful_error(cmapss_root: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download_cmapss.py"):
        load_subset(cmapss_root, "FD002")
