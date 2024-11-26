import h5py
import pandas as pd
from pathlib import Path


def save_hdf5(path: Path,
              df_d1: pd.DataFrame,
              df_d2: pd.DataFrame,
              df_d3: pd.DataFrame,
              df_coincidence: pd.DataFrame) -> None:
    data_groups = {
        'd1': df_d1,
        'd2': df_d2,
        'd3': df_d3,
        'coincidence': df_coincidence
    }

    with h5py.File(path, 'a') as hdf5_file:
        for group_name, df in data_groups.items():
            group = hdf5_file.require_group(group_name)

            # Write each column as a separate dataset within the group
            for column in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    # Convert to ns timestamp
                    data = df[column].astype('int64')
                else:
                    data = df[column].values

                dataset_path = f"{group_name}/{column}"
                # If the dataset exists, overwrite it; otherwise, create it
                if dataset_path in hdf5_file:
                    del hdf5_file[dataset_path]
                group.create_dataset(column, data=data)


def load_hdf5(path: Path):
    dataframes = {}
    column_order = ['time', 'bin', 'value', 'scaler']

    with h5py.File(path, 'r') as hdf5_file:
        for group_name in hdf5_file.keys():
            group = hdf5_file[group_name]
            data = {}

            for dataset_name in group.keys():
                dataset = group[dataset_name][:]

                # If the dataset is the 'time' column, convert it back to datetime
                if dataset_name == 'time':
                    data[dataset_name] = pd.to_datetime(dataset, unit='ns')
                else:
                    data[dataset_name] = dataset

            dataframes[group_name] = pd.DataFrame(data)
            dataframes[group_name] = dataframes[group_name][column_order]

    df_d1 = dataframes.get('d1')
    df_d2 = dataframes.get('d2')
    df_d3 = dataframes.get('d3')
    df_coincidence = dataframes.get('coincidence')

    return df_d1, df_d2, df_d3, df_coincidence
