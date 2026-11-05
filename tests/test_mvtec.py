from pathlib import Path

import pytest

from fqa.vision.mvtec import list_categories, list_samples, load_image, load_mask


def test_train_split_has_only_good_images(mvtec_root: Path) -> None:
    samples = list_samples(mvtec_root, "widget", "train")
    assert len(samples) == 2
    assert all(s.label == 0 and s.mask_path is None for s in samples)


def test_test_split_labels_and_masks(mvtec_root: Path) -> None:
    samples = list_samples(mvtec_root, "widget", "test")
    assert [s.defect_type for s in samples] == ["good", "scratch", "scratch"]
    assert [s.label for s in samples] == [0, 1, 1]
    assert samples[1].mask_path is not None and samples[1].mask_path.name == "000_mask.png"


def test_load_image_is_always_rgb(mvtec_root: Path) -> None:
    grayscale_sample = list_samples(mvtec_root, "widget", "train")[1]
    image = load_image(grayscale_sample)
    assert image.shape == (8, 8, 3)


def test_load_mask(mvtec_root: Path) -> None:
    good, defect, _ = list_samples(mvtec_root, "widget", "test")
    assert not load_mask(good).any()
    mask = load_mask(defect)
    assert mask.shape == (8, 8)
    assert mask.sum() == 4  # the 2x2 square in the fixture


def test_missing_mask_raises(mvtec_root: Path) -> None:
    (mvtec_root / "widget" / "ground_truth" / "scratch" / "001_mask.png").unlink()
    with pytest.raises(FileNotFoundError, match="Missing mask"):
        list_samples(mvtec_root, "widget", "test")


def test_missing_category_gives_helpful_error(mvtec_root: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download_mvtec.py"):
        list_samples(mvtec_root, "not_downloaded", "train")


def test_list_categories(mvtec_root: Path) -> None:
    assert list_categories(mvtec_root) == ["widget"]
