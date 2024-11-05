import h5py
import pandas as pd
from pathlib import Path

def save_hdf5(path: Path, df_p: pd.DataFrame, df_e: pd.DataFrame, df_d: pd.DataFrame, df_h: pd.DataFrame) -> None:
    data_groups = {
        'protons': df_p,
        'electrons': df_e,
        'dd': df_d,
        'heavy_ions': df_h
    }
    
    with h5py.File(path, 'a') as hdf5_file:
        for group_name, df in data_groups.items():
            group = hdf5_file.require_group(group_name)
            
            # Write each column as a separate dataset within the group
            for column in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    data = df[column].astype('int64') # Convert to ns timestamp
                else:
                    data = df[column].values

                dataset_path = f"{group_name}/{column}"
                # If the dataset exists, overwrite it; otherwise, create it
                if dataset_path in hdf5_file:
                    del hdf5_file[dataset_path]
                group.create_dataset(column, data=data)



def load_hdf5(path: Path):
    dataframes = {}

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

    df_p = dataframes.get('protons')
    df_e = dataframes.get('electrons')
    df_d = dataframes.get('dd')
    df_h = dataframes.get('heavy_ions')

    return df_p, df_e, df_d, df_h