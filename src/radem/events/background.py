import pandas as pd
import numpy as np
from scipy.stats import zscore

def intervals_from_mask(mask, signal):
    """
    Extracts contiguous intervals of True values from a mask.

    Parameters:
    - mask (pd.Series): Boolean mask where True indicates an anomaly.
    - signal (pd.DataFrame): Original signal DataFrame with a "time" column.

    Returns:
    - List of tuples: Each tuple contains (start_time, end_time) for an anomalous interval.
    """
    intervals = []
    start_idx = None

    for i, is_anomalous in enumerate(mask):
        if is_anomalous and start_idx is None:
            start_idx = i
        elif not is_anomalous and start_idx is not None:
            start_time = signal["time"].iloc[start_idx]
            end_time = signal["time"].iloc[i - 1]
            intervals.append((start_time, end_time))
            start_idx = None

    if start_idx is not None:
        start_time = signal["time"].iloc[start_idx]
        end_time = signal["time"].iloc[len(mask) - 1]
        intervals.append((start_time, end_time))

    return intervals


def autosplit_gauss_mask(background_noise, signal, window_size, threshold):
    """
    Detects anomalies in signal data based on rolling window z-score comparison with background noise.
    Creates a mask for the entire signal, marking rows as True if they belong to an anomaly.

    Parameters:
    - background_noise (pd.DataFrame): DataFrame containing background noise data in a column named "value".
    - signal (pd.DataFrame): DataFrame containing signal data with columns "time" and "value".
    - window_size (int): The size of the rolling window.
    - threshold (float): The z-score threshold to determine anomalies.

    Returns:
    - pd.Series: Boolean mask with True for rows in the signal DataFrame that are part of an anomaly.
    """

    noise_mean = background_noise["value"].mean()
    noise_std = background_noise["value"].std()

    mask = pd.Series(False, index=signal.index)

    for i in range(len(signal) - window_size + 1):
        window = signal["value"].iloc[i : i + window_size]

        window_mean = window.mean()
        z_score = (window_mean - noise_mean) / noise_std

        if abs(z_score) > threshold:
            mask.iloc[i : i + window_size] = True

    return mask

def autosplit_gauss(background_noise, signal, window_size=2*24*60, threshold=2):
    mask = autosplit_gauss_mask(background_noise, signal, window_size, threshold)
    return intervals_from_mask(mask, signal)