import pandas as pd
from pathlib import Path


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path)


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Set "time" column as index
    df.set_index('time', inplace=True)

    # Convert index to datetime[ns]
    df.index = pd.to_datetime(df.index, unit='ns')

    return df
