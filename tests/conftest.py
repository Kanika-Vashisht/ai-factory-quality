#Tiny fake datasets with the same layout as the real ones, so tests run without downloads.

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

IMG_SIZE = (8, 8)


def _save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


@pytest.fixture
def mvtec_root(tmp_path: Path) -> Path:
    #Fake MVTec category 'widget': 2 good train, 1 good test, 2 'scratch' test images.
    root = tmp_path / "mvtec"
    cat = root / "widget"
    rgb = np.full((*IMG_SIZE, 3), 128, dtype=np.uint8)
    gray = np.full(IMG_SIZE, 128, dtype=np.uint8)  # some real categories are grayscale

    _save_png(cat / "train" / "good" / "000.png", rgb)
    _save_png(cat / "train" / "good" / "001.png", gray)
    _save_png(cat / "test" / "good" / "000.png", rgb)
    for i in range(2):
        _save_png(cat / "test" / "scratch" / f"00{i}.png", rgb)
        mask = np.zeros(IMG_SIZE, dtype=np.uint8)
        mask[2:4, 2:4] = 255
        _save_png(cat / "ground_truth" / "scratch" / f"00{i}_mask.png", mask)
    return root


def _write_rows(path: Path, rows: list[list[float]]) -> None:
    # Mimic the raw files: space separated with a trailing space on every line.
    path.write_text("".join(" ".join(str(v) for v in row) + "  \n" for row in rows))


def _engine_rows(unit: int, n_cycles: int) -> list[list[float]]:
    return [[unit, c, 0.0, 0.0, 100.0, *([1.0] * 21)] for c in range(1, n_cycles + 1)]


@pytest.fixture
def cmapss_root(tmp_path: Path) -> Path:
    #Fake FD001: train engines run 5 and 3 cycles; test engines stop after 2 and 4 cycles.
    root = tmp_path / "cmapss"
    root.mkdir()
    _write_rows(root / "train_FD001.txt", _engine_rows(1, 5) + _engine_rows(2, 3))
    _write_rows(root / "test_FD001.txt", _engine_rows(1, 2) + _engine_rows(2, 4))
    (root / "RUL_FD001.txt").write_text("10 \n7 \n")
    return root
