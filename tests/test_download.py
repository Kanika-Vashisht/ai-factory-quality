import tarfile
from pathlib import Path

import pytest

from fqa.download import extract_tar, sha256, verify_sha256


def test_sha256_known_value(tmp_path: Path) -> None:
    path = tmp_path / "hello.txt"
    path.write_bytes(b"hello")
    expected = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert sha256(path) == expected
    verify_sha256(path, expected)


def test_verify_sha256_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "hello.txt"
    path.write_bytes(b"hello")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_sha256(path, "0" * 64)


def test_extract_tar_blocks_path_traversal(tmp_path: Path) -> None:
    payload = tmp_path / "evil.txt"
    payload.write_text("x")
    archive = tmp_path / "evil.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(payload, arcname="../escaped.txt")

    with pytest.raises(tarfile.TarError):
        extract_tar(archive, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()
