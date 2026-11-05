#to load the NASA C-MAPSS turbofan degradation data

# train_FDxxx.txt   full run-to-failure trajectories (the last cycle is the failure)
# test_FDxxx.txt    trajectories cut off at some point before failure
# RUL_FDxxx.txt     true remaining cycles for each test engine at its last row

#every row is one engine at one cycle: unit id, cycle, 3 operational settings, 21 sensors.


from pathlib import Path
from typing import Literal

import pandas as pd

Subset = Literal["FD001", "FD002", "FD003", "FD004"]
SUBSETS: tuple[Subset, ...] = ("FD001", "FD002", "FD003", "FD004")

SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTING_COLS, *SENSOR_COLS]


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/download_cmapss.py`.")
    # sep=r"\s+" handles the trailing spaces at the end of every line in the raw files.
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    return df.astype({"unit": int, "cycle": int})


def add_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    #Training engines run until failure, so RUL = last cycle of that engine - current cycle.
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    return df.assign(rul=last_cycle - df["cycle"])


def add_test_rul(df: pd.DataFrame, final_rul: pd.Series) -> pd.DataFrame:
    #Test engines stop early. final_rul[unit] is the RUL at their last row
    # so RUL at any row = final_rul + (last cycle - current cycle)
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    return df.assign(rul=df["unit"].map(final_rul) + last_cycle - df["cycle"])


def load_subset(root: Path, subset: Subset) -> tuple[pd.DataFrame, pd.DataFrame]:
    #Return (train, test) DataFrames for one subset, both with an rul column
    train = add_train_rul(_read_table(root / f"train_{subset}.txt"))
    test = _read_table(root / f"test_{subset}.txt")

    rul_path = root / f"RUL_{subset}.txt"
    if not rul_path.exists():
        raise FileNotFoundError(f"{rul_path} not found. Run `python scripts/download_cmapss.py`.")
    final_rul = pd.read_csv(rul_path, sep=r"\s+", header=None).iloc[:, 0]
    # Line i of the RUL file belongs to test unit i + 1.
    final_rul.index = range(1, len(final_rul) + 1)

    n_units = test["unit"].nunique()
    if len(final_rul) != n_units:
        raise ValueError(f"{rul_path} has {len(final_rul)} rows but test has {n_units} units")

    return train, add_test_rul(test, final_rul)
