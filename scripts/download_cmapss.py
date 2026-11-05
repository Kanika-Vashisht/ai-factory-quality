import io
import zipfile

from fqa.config import data_dir
from fqa.download import download_file, verify_sha256

URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
SHA256 = "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"
INNER_ZIP = "6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip"


def main() -> None:
    out_dir = data_dir() / "cmapss"
    if (out_dir / "train_FD001.txt").exists():
        print(f"[skip] C-MAPSS already in {out_dir}")
        return

    archive = download_file(URL, data_dir() / "archives" / "cmapss.zip")
    verify_sha256(archive, SHA256)

    with zipfile.ZipFile(archive) as outer:
        inner_bytes = outer.read(INNER_ZIP)
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        inner.extractall(out_dir)

    archive.unlink()
    print(f"Done. Files in {out_dir}:")
    for path in sorted(out_dir.iterdir()):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
