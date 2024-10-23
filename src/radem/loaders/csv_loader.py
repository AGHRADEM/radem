from pathlib import Path
import pandas as pd

def load_csv(filename: str) -> pd.DataFrame:
    """
    Load a CSV file with RADEM science data into a pandas DataFrame and validate its structure.

    Parameters:
    ----------
    filename : str
        The path to the CSV file to be loaded.

    Returns:
    -------
    pd.DataFrame
        A pandas DataFrame containing the data.

    Raises:
    ------
    FileNotFoundError
        If the specified file does not exist.
    KeyError
        If any of the expected columns ('time', 'bin', 'value') are missing from the DataFrame.
    TypeError
        If the 'time' column cannot be parsed as datetime or if any columns do not match the expected data types.
    """

    df = pd.read_csv(filename)
    df['time'] = pd.to_datetime(df['time'])

    expected_columns = {
        'time': 'datetime64[ns]',
        'bin': 'int',
        'value': 'int'
    }

    for column in expected_columns.keys():
        if column not in df.columns:
            raise KeyError(f"Missing column: {column}")

    for column, expected_type in expected_columns.items():
        if column == 'time':
            if not pd.to_datetime(df[column], errors='coerce').notnull().all():
                raise TypeError(f"Column '{column}' is not of type {expected_type}.")
        elif not pd.api.types.is_dtype_equal(df[column].dtype, expected_type):
            raise TypeError(f"Column '{column}' is not of type {expected_type}.")

    return df

def save_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Save the CSV data to a specified path.
    """
    df.to_csv(path, index=False)