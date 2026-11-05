#index and load images from the MVTec AD dataset
# train/good/000.png                 only defect-free images
# test/good/000.png                  defect-free test images
# test/<defect_type>/000.png         defective test images
# ground_truth/<defect_type>/000_mask.png   pixel mask for each defective image

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

Split = Literal["train", "test"]
GOOD = "good"


@dataclass(frozen=True)
class MVTecSample:
    image_path: Path
    category: str
    defect_type: str  # "good" or the defect name, e.g. "scratch_head"
    mask_path: Path | None  # None for good images (their mask is all zeros)

    @property
    def label(self) -> int:
        #1 = defective, 0 = good
        return int(self.defect_type != GOOD)


def list_samples(root: Path, category: str, split: Split) -> list[MVTecSample]:
    split_dir = root / category / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"{split_dir} not found. "
            f"Run `python scripts/download_mvtec.py --categories {category}`."
        )

    samples: list[MVTecSample] = []
    for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        defect_type = defect_dir.name
        for image_path in sorted(defect_dir.glob("*.png")):
            mask_path = None
            if defect_type != GOOD:
                mask_path = (
                    root / category / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
                )
                if not mask_path.exists():
                    raise FileNotFoundError(f"Missing mask for {image_path}: {mask_path}")
            samples.append(MVTecSample(image_path, category, defect_type, mask_path))
    return samples


def list_categories(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if (p / "train").is_dir())


def load_image(sample: MVTecSample) -> np.ndarray:
    #Loading an image as an RGB uint8 array of shape (H, W, 3) so that every model has the same input shape.

    #since some categories (grid, screw, zipper) are grayscale; 
    with Image.open(sample.image_path) as img:
        return np.asarray(img.convert("RGB"))


def load_mask(sample: MVTecSample, size: tuple[int, int] | None = None) -> np.ndarray:
    #loading the defect mask as a bool array of shape (H, W). Good images get an all-False mask.
    #size is only needed for good images, which have no mask file;
    
    if sample.mask_path is not None:
        with Image.open(sample.mask_path) as mask:
            return np.asarray(mask.convert("L")) > 0
    if size is None:
        with Image.open(sample.image_path) as img:
            size = (img.height, img.width)
    return np.zeros(size, dtype=bool)
