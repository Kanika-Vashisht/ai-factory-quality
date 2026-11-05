#Source: https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads
#License: CC BY-NC-SA 4.0 (non-commercial use only).
import argparse

from fqa.config import data_dir, load_config
from fqa.download import download_file, extract_tar

# Per-category links from the official MVTec download page (whole dataset is 4.9 GB).
URLS: dict[str, str] = {
    "bottle": "https://www.mydrive.ch/shares/163640/9a565b9a02f45cd135e46b1c36e01ec5/download/420937370-1629958698/bottle.tar.xz",
    "cable": "https://www.mydrive.ch/shares/163641/97fd0295152453e533cfb3db1d253a35/download/420937413-1629958794/cable.tar.xz",
    "capsule": "https://www.mydrive.ch/shares/163642/236d4304eaf9526b350b18807789ce86/download/420937454-1629958872/capsule.tar.xz",
    "carpet": "https://www.mydrive.ch/shares/163643/70e8bebbe9f2d2c55043311a4e67eb1d/download/420937484-1629959013/carpet.tar.xz",
    "grid": "https://www.mydrive.ch/shares/163644/8c7fc7cba43ef5275c87ca750a6d0004/download/420937487-1629959044/grid.tar.xz",
    "hazelnut": "https://www.mydrive.ch/shares/163645/8346f81c9e093d2ebbb3a73f98372217/download/420937545-1629959162/hazelnut.tar.xz",
    "leather": "https://www.mydrive.ch/shares/163646/7e0627b5718d2902bb85d24cff7a51ce/download/420937607-1629959262/leather.tar.xz",
    "metal_nut": "https://www.mydrive.ch/shares/163647/6d5a833cdee9035a45d5ce0921ecd24d/download/420937637-1629959294/metal_nut.tar.xz",
    "pill": "https://www.mydrive.ch/shares/163648/f8c4ca0d325888f2ba128a9ed44708f5/download/420938129-1629960351/pill.tar.xz",
    "screw": "https://www.mydrive.ch/shares/163649/cfd394b4e3a49013745fab9e2084988a/download/420938130-1629960389/screw.tar.xz",
    "tile": "https://www.mydrive.ch/shares/163650/55ad2038770deff6fe6771bd784af8cc/download/420938133-1629960456/tile.tar.xz",
    "toothbrush": "https://www.mydrive.ch/shares/163651/f03d8421ec64b3e3bfa4a2ba2fbafb4f/download/420938134-1629960477/toothbrush.tar.xz",
    "transistor": "https://www.mydrive.ch/shares/163652/619a88c3d5f3f57b46b00f44f04ad6e6/download/420938166-1629960554/transistor.tar.xz",
    "wood": "https://www.mydrive.ch/shares/163653/937e0d65bc9c4497d253cb7313868a60/download/420938383-1629960649/wood.tar.xz",
    "zipper": "https://www.mydrive.ch/shares/163654/7f6e6aefbf00a9e8a09cc0bbac236eb0/download/420938385-1629960680/zipper.tar.xz",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--categories", nargs="+", choices=sorted(URLS), default=None)
    parser.add_argument(
        "--keep-archives", action="store_true", help="keep .tar.xz files after extraction"
    )
    args = parser.parse_args()

    categories = args.categories or load_config("data")["mvtec"]["categories"]
    out_dir = data_dir() / "mvtec"
    archive_dir = data_dir() / "archives"

    for category in categories:
        if (out_dir / category / "train").is_dir():
            print(f"[skip] {category} already extracted")
            continue
        archive = download_file(URLS[category], archive_dir / f"{category}.tar.xz")
        print(f"Extracting {archive.name} ...")
        extract_tar(archive, out_dir)  # archive contains a top-level <category>/ folder
        if not args.keep_archives:
            archive.unlink()

    print(f"Done. Data in {out_dir}")


if __name__ == "__main__":
    main()
