import pandas as pd
import numpy as np

from spacepy import pycdf
from pathlib import Path
from datetime import datetime, date
from typing import List, Tuple, Optional


def filename_to_date(filename: str | Path) -> date:
    filename = str(filename.name)

    date_str = filename[10:18]
    file_date = datetime.strptime(date_str, "%Y%m%d").date()
    return file_date


def is_filename_after_date(filename: str,
                           date_filter: date) -> bool:
    file_date = filename_to_date(filename)
    return file_date >= date_filter


def is_filename_before_date(filename: str,
                            date_filter: date) -> bool:
    file_date = filename_to_date(filename)
    return file_date < date_filter


def read_cdf(cdf_path: Path) -> Optional[pycdf.CDF]:
    if not cdf_path.stat().st_size:
        return None
    return pycdf.CDF(str(cdf_path), readonly=True)


def read_irem_cdfs(data_dir: Path,
                   from_date: Optional[date] = None,
                   to_date: Optional[date] = None) -> List[pycdf.CDF]:
    cdfs = []
    for filename in sorted(data_dir.glob("*.cdf")):
        if from_date and is_filename_before_date(filename, from_date):
            continue

        if to_date and is_filename_after_date(filename, to_date):
            continue

        cdf = read_cdf(filename)
        print(filename)
        if cdf:
            cdfs.append(cdf)
    return cdfs


def standardize_irem_df(df: pd.DataFrame) -> None:
    df['time'] = pd.to_datetime(df['time']).dt.floor('s')
    df.drop_duplicates(inplace=True)
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)


def process_irem_particles(cdf: pycdf.CDF,
                           scaler_idx_start: int,
                           scaler_idx_end: int) -> pd.DataFrame:
    # According do the IREM User Manual:
    # CDF label_COUNTERS always are:
    #   - [TC1 S12 S13 S14 S15 TC2 S25 C1 C2 C3 C4 TC3 S32 S33 S34]
    #   - and are 3 characters long e.g. "C1 "
    n = len(cdf["EPOCH"][...])
    times = cdf["EPOCH"][...]

    d_scalers = cdf["label_COUNTERS"][..., scaler_idx_start:scaler_idx_end]
    d = cdf["COUNTRATE"][..., scaler_idx_start:scaler_idx_end]

    time_col = np.repeat(times, len(d_scalers))
    scaler_col = np.tile(d_scalers, n)
    value_col = d.flatten()
    bin_col = np.tile(np.arange(1, 1 + len(d_scalers)), n)

    df = pd.DataFrame({
        "time": time_col,
        "scaler": scaler_col,
        "value": value_col,
        "bin": bin_col
    })

    return df


def process_irem_d1(cdf: pycdf.CDF) -> pd.DataFrame:
    # According do the IREM User Manual:
    # label_COUNTERS[0:5] is [TC1 S12 S13 S14 S15] which is D1
    return process_irem_particles(cdf, 0, 5)


def process_irem_d2(cdf: pycdf.CDF) -> pd.DataFrame:
    # According do the IREM User Manual:
    # label_COUNTERS[5:7] is [TC2 S25] which is D2
    return process_irem_particles(cdf, 5, 7)


def process_irem_coincidence(cdf: pycdf.CDF) -> pd.DataFrame:
    # According do the IREM User Manual:
    # label_COUNTERS[7:11] is [C1 C2 C3 C4] which is D1+D2 Coincidence
    return process_irem_particles(cdf, 7, 11)


def process_irem_d3(cdf: pycdf.CDF) -> pd.DataFrame:
    # According do the IREM User Manual:
    # label_COUNTERS[11:15] is [TC3 S32 S33 S34] which is D3
    return process_irem_particles(cdf, 11, 15)


def irem_cdf_to_df(cdfs: List[pycdf.CDF], process_irem_fn) -> pd.DataFrame:
    df = pd.concat(process_irem_fn(cdf) for cdf in cdfs)
    standardize_irem_df(df)
    return df


def load_science_cdfs(data_dir: Path,
                      from_date: Optional[date] = None,
                      to_date: Optional[date] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    science_cdfs = read_irem_cdfs(data_dir, from_date, to_date)

    df_d1 = irem_cdf_to_df(science_cdfs, process_irem_d1)
    df_d2 = irem_cdf_to_df(science_cdfs, process_irem_d2)
    df_d3 = irem_cdf_to_df(science_cdfs, process_irem_d3)
    df_coincidence = irem_cdf_to_df(science_cdfs, process_irem_coincidence)

    del science_cdfs

    return df_d1, df_d2, df_d3, df_coincidence
