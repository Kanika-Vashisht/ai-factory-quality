#helpers for reading YAML config files

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_config(name: str) -> dict[str, Any]:
    #loading `configs/<name>.yaml` as a dict
    path = CONFIG_DIR / f"{name}.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_dir() -> Path:
    return PROJECT_ROOT / load_config("data")["data_dir"]
