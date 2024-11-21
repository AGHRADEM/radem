import pandas as pd
import numpy as np

from spacepy import pycdf
from pathlib import Path
from typing import List, Tuple


def read_science_cdfs(data_dir: Path) -> List[pycdf.CDF]:
    """
    Reads all science CDF files in the specified directory structure.

    This function navigates through the `data_dir`, locating subdirectories
    with science data CDF files (filenames starting with "rad_raw_sc_"). It
    then loads each matching CDF file and returns a list of CDF objects.

    Args:
        data_dir (Path): The root directory containing batch directories
            with CDF files in the "juice_rad/data_raw" subdirectory.

    Returns:
        List[pycdf.CDF]: A list of CDF objects representing the science data files.
    """

    def is_path_science_cdf(path: Path) -> bool:
        return path.name.startswith("rad_raw_sc_") and path.name.endswith(".cdf")

    cdfs = []
    for batch_dir in sorted(data_dir.iterdir()):
        cdf_dir = batch_dir.joinpath("juice_rad/data_raw")
        for cdf_path in cdf_dir.glob("*.cdf"):
            if is_path_science_cdf(cdf_path):
                cdfs.append(pycdf.CDF(str(cdf_path)))
    return cdfs


def read_housekeeping_cdfs(data_dir: Path) -> List[pycdf.CDF]:
    """
    Reads all housekeeping CDF files in the specified directory structure.

    This function navigates through the `data_dir`, locating subdirectories
    with housekeeping data CDF files (filenames starting with "rad_raw_hk_"). It
    then loads each matching CDF file and returns a list of CDF objects.

    Args:
        data_dir (Path): The root directory containing batch directories
            with CDF files in the "juice_rad/data_raw" subdirectory.

    Returns:
        List[pycdf.CDF]: A list of CDF objects representing the housekeeping data files.
    """
    def is_path_housekeeping_cdf(path: Path) -> bool:
        return path.name.startswith("rad_raw_hk_") and path.name.endswith(".cdf")

    cdfs = []
    for batch_dir in sorted(data_dir.iterdir()):
        cdf_dir = batch_dir.joinpath("juice_rad/data_raw")
        for cdf_path in cdf_dir.glob("*.cdf"):
            if is_path_housekeeping_cdf(cdf_path):
                cdfs.append(pycdf.CDF(str(cdf_path)))
    return cdfs


def process_particles(cdf: pycdf.CDF, cdf_particle_key: str, cdf_particle_bin: str) -> pd.DataFrame:
    times = cdf["TIME_UTC"][...]
    particles = cdf[cdf_particle_key][...]
    particle_bins = cdf[cdf_particle_bin][...]

    time_col = np.repeat(times, len(particle_bins))
    bin_col = np.tile(particle_bins, len(particles))
    value_col = particles.flatten()

    df = pd.DataFrame({
        "time": time_col,
        "bin": bin_col,
        "value": value_col
    })

    return df


def convert_housekeeping_temp(adc_out: int | np.ndarray) -> np.ndarray:
    """
    Converts housekeeping temperature (ADC output) to Celsius.

    This function uses Equation 6 for RADEM EQM/PFM HK from the RADEM User Manual
    to convert the ADC output to temperature in Celsius. It is applicable for
    temperature sensors 1-5 and provides a precision of 1 Celsius degree. The
    RADEM operating range is from -40 to +85 Celsius degrees.

    Args:
        adc_out (int | np.ndarray): The ADC output value(s) to be converted.

    Returns:
        np.ndarray: The converted temperatures in Celsius.
    """
    return np.round(adc_out * (3.3 / 4096) * (1000000 / 2210) - 273.16)


def process_temperatures(cdf: pycdf.CDF) -> pd.DataFrame:
    TEMP_KEYS = [
        "HK_Temp1_CEU",
        "HK_PandI_Stack_Temp2",
        "HK_E_Stack_Temp3",
        "HK_DD_Temp4",
        "HK_Temp5_CPU",
    ]

    DF_KEYS = [
        "time",
        "CEU Temperature (1)",
        "P&IDH Temperature (2)",
        "EDH Temperature (3)",
        "DDH Temperature (4)",
        "PCU Temperature (5)",
    ]

    time_column = cdf["TIME_UTC"][...]
    temp_columns = [convert_housekeeping_temp(
        cdf[key][...]) for key in TEMP_KEYS]
    data = np.column_stack([time_column, *temp_columns])
    return pd.DataFrame(data, columns=DF_KEYS)


def standardize_df(df: pd.DataFrame) -> None:
    """
    Cleans and standardizes a DataFrame by ensuring time column sorting, removing duplicates,
    and filtering for relevant data entries. These operations are performed in place.

    This function performs the following operations on the input DataFrame:
    - Converts the 'time' column to a datetime format with seconds precision.
    - Filters the data to include only entries from September 1, 2023, onward.
    - Removes any duplicate rows, keeping only the first occurrence.
    - Sorts the DataFrame by the 'time' column in ascending order.

    Args:
        df: A DataFrame with RADEM data

    Returns: The cleaned DataFrame with sorted, unique, and relevant data.
    """
    df["time"] = pd.to_datetime(df['time']).dt.floor('s')
    df.query("time >= '2023-09-01'", inplace=True)
    df.drop_duplicates(inplace=True, keep="first")
    df.sort_values("time", inplace=True)


def load_science_cdfs(directory: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads and processes science CDF files, returning DataFrames for particle data.

    This function reads all science CDF files from the specified directory, processes
    the particle data by particle type, and returns standardized DataFrames for:
    - Proton Stack Detector (PSD) data
    - Electron Detector Head (EDH) data
    - DD (Directional Detector) data
    - High Ions Detector Head (HIDH) data

    Each DataFrame is concatenated across all files, and is standardized to ensure consistent
    formatting.

    Args:
        directory  The path to the directory containing science CDF files.

    Returns: A tuple containing four DataFrames in the following order:
            - df_p: DataFrame with proton data
            - df_e: DataFrame with electron data
            - df_d: DataFrame with DD particle data
            - df_h: DataFrame with high-ion particle data
    """
    science_cdfs = read_science_cdfs(directory)

    df_p = pd.concat((
        process_particles(cdf, "PROTONS", "PROTON_BINS")
        for cdf in science_cdfs
    ))

    df_e = pd.concat((
        process_particles(cdf, "ELECTRONS", "ELECTRON_BINS")
        for cdf in science_cdfs
    ))

    df_d = pd.concat((
        process_particles(cdf, "DD", "DD_BINS")
        for cdf in science_cdfs
    ))

    df_h = pd.concat((
        process_particles(cdf, "HI_IONS", "HI_ION_BINS")
        for cdf in science_cdfs
    ))

    standardize_df(df_p)
    standardize_df(df_e)
    standardize_df(df_d)
    standardize_df(df_h)

    del science_cdfs

    return df_p, df_e, df_d, df_h


def load_housekeeping_cdfs(directory: Path) -> pd.DataFrame:
    """
    Loads and processes housekeeping CDF files, returning a DataFrame for temperature data.

    This function reads all housekeeping CDF files from the specified directory, processes
    the temperature data, and returns a standardized DataFrame.

    The DataFrame is concatenated across all files, and is standardized to ensure consistent
    formatting.

    Args:
        directory  The path to the directory containing science CDF files.

    Returns: A DataFrame with temperature data.
    """
    housekeeping_cdfs = read_housekeeping_cdfs(directory)

    df_temp = pd.concat((
        process_temperatures(cdf)
        for cdf in housekeeping_cdfs
    ))

    standardize_df(df_temp)

    del housekeeping_cdfs

    return df_temp
