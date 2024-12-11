import pandas as pd
from pathlib import Path

_HDF5_KEY = "df"


def write_hdf(df: pd.DataFrame, path: Path) -> None:
    df.to_hdf(path,
              key=_HDF5_KEY,
              mode="a",
              format='table')


def append_hdf(df: pd.DataFrame, path: Path) -> None:
    df.to_hdf(path,
              key=_HDF5_KEY,
              mode="a",
              format='table',
              append=True)


def read_hdf(path: Path) -> pd.DataFrame:
    return pd.read_hdf(path, key=_HDF5_KEY)
