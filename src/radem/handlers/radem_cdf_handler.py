import spacepy.pycdf as pycdf
import pandas as pd
import numpy as np
import re

from pathlib import Path
from typing import List, Optional
from datetime import date, datetime


def get_radem_science_cdf_paths(data_dir: Path,
                                from_date: Optional[date] = None,
                                to_date: Optional[date] = None) -> List[Path]:
    # NOTE: from_date and to_date are used but the date inside file might be
    #       different so we need to check if the file is in the date range

    path_generator = data_dir.rglob("*.cdf")
    paths = [path for path in path_generator
             if _is_radem_science_cdf_path_valid(path, from_date, to_date)]
    paths.sort()
    return paths


def get_radem_housekeeping_cdf_paths(data_dir: Path,
                                     from_date: Optional[date] = None,
                                     to_date: Optional[date] = None) \
        -> List[Path]:
    path_generator = data_dir.rglob("*.cdf")
    paths = [path for path in path_generator
             if _is_radem_housekeeping_cdf_path_valid(path,
                                                      from_date,
                                                      to_date)]
    paths.sort()
    return paths


def read_radem_cdf(path: Path) -> pycdf.CDF:
    return pycdf.CDF(str(path), readonly=True)


def read_radem_cdfs(paths: List[Path]) -> List[pycdf.CDF]:
    return [read_radem_cdf(path) for path in paths]


def read_radem_science_cdfs(data_dir: Path,
                            from_date: Optional[date] = None,
                            to_date: Optional[date] = None) -> List[pycdf.CDF]:
    paths = get_radem_science_cdf_paths(data_dir, from_date, to_date)
    return read_radem_cdfs(paths)


def read_radem_housekeeping_cdfs(data_dir: Path,
                                 from_date: Optional[date] = None,
                                 to_date: Optional[date] = None) -> List[pycdf.CDF]:
    paths = get_radem_housekeeping_cdf_paths(data_dir, from_date, to_date)
    return read_radem_cdfs(paths)


def convert_radem_science_cdf_to_df(cdf: pycdf.CDF) -> pd.DataFrame:
    NO_OF_PROTON_BINS = 8
    NO_OF_ELECTRON_BINS = 8
    NO_OF_DIRECTIONAL_BINS = 31
    NO_OF_HI_ION_BINS = 7

    df = pd.DataFrame({
        "time": cdf["TIME_UTC"][...],
        **{
            f"protons_bin_{i+1}": cdf["PROTONS"][..., i]
            for i in range(NO_OF_PROTON_BINS)
        },
        "proton_bin_others": cdf["PROTONS"][..., NO_OF_PROTON_BINS],
        **{
            f"electrons_bin_{i+1}": cdf["ELECTRONS"][..., i]
            for i in range(NO_OF_ELECTRON_BINS)
        },
        "electron_bin_others": cdf["ELECTRONS"][..., NO_OF_ELECTRON_BINS],
        **{
            f"directional_bin_{i+1}": cdf["DD"][..., i]
            for i in range(NO_OF_DIRECTIONAL_BINS)
        },
        **{
            f"hi_ion_bin_{i+1}": cdf["HI_IONS"][..., i]
            for i in range(NO_OF_HI_ION_BINS)
        },
        "hi_ion_bin_others": cdf["HI_IONS"][..., NO_OF_HI_ION_BINS],
        "flux_protons": cdf["FLUX"][..., 0],
        "flux_electrons": cdf["FLUX"][..., 1],
        "flux_directional": cdf["FLUX"][..., 2],
    })

    # Convert time to datetime and floor to seconds
    df["time"] = pd.to_datetime(df['time']).dt.floor('s')

    # Drop data before 2023-09-01, it hasn't scientific value
    df.query("time >= '2023-09-01'", inplace=True)

    # Raw CDFs might contain duplicates, so we need to drop them
    df.drop_duplicates(inplace=True)

    # We need to sort after dropping duplicates
    df.sort_values('time', inplace=True)

    # Set time as index
    df.set_index('time', inplace=True)

    return df


def _convert_housekeeping_temp(adc_out: np.ndarray) -> np.ndarray:
    return np.round(adc_out * (3.3 / 4096) * (1000000 / 2210) - 273.16)


def convert_radem_housekeeping_cdf_to_df(cdf: pycdf.CDF) -> pd.DataFrame:
    df = pd.DataFrame({
        "time": cdf["TIME_UTC"][...],
        "ceu_temp_celsius":
            _convert_housekeeping_temp(cdf["HK_Temp1_CEU"][...]),
        "protons_and_hi_ions_temp_celsius":
            _convert_housekeeping_temp(cdf["HK_PandI_Stack_Temp2"][...]),
        "electrons_temp_celsius":
            _convert_housekeeping_temp(cdf["HK_E_Stack_Temp3"][...]),
        "directionals_temp_celsius":
            _convert_housekeeping_temp(cdf["HK_DD_Temp4"][...]),
        "pcu_temp_celsius":
            _convert_housekeeping_temp(cdf["HK_Temp5_CPU"][...]),
    })
    return df


def convert_radem_science_cdfs_to_df(cdfs: List[pycdf.CDF]) -> pd.DataFrame:
    df = pd.concat([convert_radem_science_cdf_to_df(cdf) for cdf in cdfs])
    return df


def _convert_radem_cdf_path_to_date(path: Path) -> date:
    path = str(path.name)
    date_str = path[11:19]
    file_date = datetime.strptime(date_str, "%Y%m%d").date()
    return file_date


def _is_radem_cdf_path_in_date_range(path: Path,
                                     from_date: Optional[date] = None,
                                     to_date: Optional[date] = None) -> bool:
    file_date = _convert_radem_cdf_path_to_date(path)
    if from_date and file_date < from_date:
        return False
    if to_date and file_date >= to_date:
        return False
    return True


def _is_radem_science_cdf_path_naming_correct(path: Path) -> bool:
    pattern = re.compile(r"rad_raw_sc_\d{8}.cdf")
    return pattern.match(path.name)


def _is_radem_housekeeping_cdf_path_naming_correct(path: Path) -> bool:
    pattern = re.compile(r"rad_raw_hk_\d{8}.cdf")
    return pattern.match(path.name)


def _is_path_existing(path: Path) -> bool:
    return path.exists()


def _is_path_empty(path: Path) -> bool:
    return path.stat().st_size == 0


def _is_radem_cdf_path_valid(path: Path,
                             from_date: Optional[date] = None,
                             to_date: Optional[date] = None) -> bool:
    if not _is_path_existing(path):
        return False
    if _is_path_empty(path):
        return False
    if not _is_radem_cdf_path_in_date_range(path, from_date, to_date):
        return False
    return True


def _is_radem_science_cdf_path_valid(path: Path,
                                     from_date: Optional[date] = None,
                                     to_date: Optional[date] = None) -> bool:
    if not _is_radem_science_cdf_path_naming_correct(path):
        return False
    if not _is_radem_cdf_path_valid(path, from_date, to_date):
        return False
    return True


def _is_radem_housekeeping_cdf_path_valid(path: Path,
                                          from_date: Optional[date] = None,
                                          to_date: Optional[date] = None) \
        -> bool:
    if not _is_radem_housekeeping_cdf_path_naming_correct(path):
        return False
    if not _is_radem_cdf_path_valid(path, from_date, to_date):
        return False
    return True
