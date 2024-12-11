import spacepy.pycdf as pycdf
import pandas as pd
import re

from pathlib import Path
from typing import List, Optional, overload, Union
from datetime import date, datetime


def get_irem_cdf_paths(data_dir: Path,
                       from_date: Optional[date] = None,
                       to_date: Optional[date] = None) -> List[Path]:
    path_generator = data_dir.glob("*.cdf")
    paths = [path for path in path_generator
             if _is_irem_cdf_path_valid(path, from_date, to_date)]
    paths.sort()
    return paths


def read_irem_cdf(path: Path) -> pycdf.CDF:
    return pycdf.CDF(str(path), readonly=True)


@overload
def read_irem_cdfs(paths: List[Path]) -> List[pycdf.CDF]:
    ...


@overload
def read_irem_cdfs(data_dir: Path,
                   from_date: Optional[date] = None,
                   to_date: Optional[date] = None) -> List[pycdf.CDF]:
    ...


def read_irem_cdfs(arg0: Union[List[Path], Path],
                   from_date: Optional[date] = None,
                   to_date: Optional[date] = None) -> List[pycdf.CDF]:
    if isinstance(arg0, list):
        paths = arg0
    else:
        paths = get_irem_cdf_paths(arg0, from_date, to_date)
    return [read_irem_cdf(path) for path in paths]


def convert_irem_cdf_to_df(cdf: pycdf.CDF) -> pd.DataFrame:
    # According do the IREM User Manual:
    # * label_COUNTERS[0:5] is [TC1 S12 S13 S14 S15] which is D1
    # * label_COUNTERS[5:7] is [TC2 S25] which is D2
    # * label_COUNTERS[7:11] is [C1 C2 C3 C4] which is D1+D2 Coincidence
    # * label_COUNTERS[11:15] is [TC3 S32 S33 S34] which is D3
    df = pd.DataFrame({
        "time": cdf["EPOCH"][...],
        "d1_channel1":  cdf["COUNTRATE"][..., 0],
        "d1_channel2":  cdf["COUNTRATE"][..., 1],
        "d1_channel3":  cdf["COUNTRATE"][..., 2],
        "d1_channel4":  cdf["COUNTRATE"][..., 3],
        "d1_channel5":  cdf["COUNTRATE"][..., 4],
        "d2_channel1":  cdf["COUNTRATE"][..., 5],
        "d2_channel2":  cdf["COUNTRATE"][..., 6],
        "coincidence_channel1":   cdf["COUNTRATE"][..., 7],
        "coincidence_channel2":   cdf["COUNTRATE"][..., 8],
        "coincidence_channel3":   cdf["COUNTRATE"][..., 9],
        "coincidence_channel4":   cdf["COUNTRATE"][..., 10],
        "d3_channel1":  cdf["COUNTRATE"][..., 11],
        "d3_channel2":  cdf["COUNTRATE"][..., 12],
        "d3_channel3":  cdf["COUNTRATE"][..., 13],
        "d3_channel4":  cdf["COUNTRATE"][..., 14],
    })

    # Raw CDFs might contain duplicates, so we need to drop them
    df.drop_duplicates(inplace=True)

    # We need to sort after dropping duplicates
    df.sort_values('time', inplace=True)

    # Set time as index
    df.set_index('time', inplace=True)

    return df


def convert_irem_cdfs_to_df(cdfs: List[pycdf.CDF]) -> pd.DataFrame:
    df = pd.concat([convert_irem_cdf_to_df(cdf) for cdf in cdfs])
    return df


def _convert_irem_cdf_path_to_date(path: Path) -> date:
    path = str(path.name)
    date_str = path[10:18]
    file_date = datetime.strptime(date_str, "%Y%m%d").date()
    return file_date


def _is_irem_cdf_path_in_date_range(path: Path,
                                    from_date: Optional[date] = None,
                                    to_date: Optional[date] = None) -> bool:
    file_date = _convert_irem_cdf_path_to_date(path)
    if from_date and file_date < from_date:
        return False
    if to_date and file_date >= to_date:
        return False
    return True


def _is_irem_cdf_path_naming_correct(path: Path) -> bool:
    # NOTE: Be aware that there are a couple of files what don't follow
    # this convention and can introduce data duplicates e.g.:
    # * IREM_PACC_20030128_exp.cdf
    # * IREM_PACC_20030128_pow.cdf
    # * IREM_PACC_20030128.cdf
    # and others
    pattern = re.compile(r"IREM_PACC_\d{8}\.cdf")
    return pattern.match(path.name)


def _is_path_existing(path: Path) -> bool:
    return path.exists()


def _is_path_empty(path: Path) -> bool:
    return path.stat().st_size == 0


def _is_irem_cdf_path_valid(path: Path,
                            from_date: Optional[date] = None,
                            to_date: Optional[date] = None) -> bool:
    if not _is_irem_cdf_path_naming_correct(path):
        return False
    if not _is_path_existing(path):
        return False
    if _is_path_empty(path):
        return False
    if not _is_irem_cdf_path_in_date_range(path, from_date, to_date):
        return False
    return True
