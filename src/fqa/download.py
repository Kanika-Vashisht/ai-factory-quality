#Download and extract helpers shared by the scripts in scripts

import hashlib
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> Path:
    
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with (
            tmp.open("wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
        ):
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)
    return dest


def sha256(path: Path) -> str:
    #the SHA-256 hex digest of a file.
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    #error if the file's checksum does not match `expected`
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {path}: expected {expected}, got {actual}")


def extract_tar(archive: Path, dest: Path) -> None:
    #extract a .tar/.tar.xz archive. The 'data' filter blocks path traversal and links
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
