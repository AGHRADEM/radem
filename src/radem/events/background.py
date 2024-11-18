import pandas as pd
import numpy as np
from scipy.stats import zscore

def autosplit_gauss(background_noise, signal, window_size=50, threshold=3):
    """
    Detects anomalies in signal data based on rolling window z-score comparison with background noise.
    Marks the entire window as anomalous if an anomaly is detected.

    Parameters:
    - background_noise (pd.DataFrame): DataFrame containing background noise data in a column named "value".
    - signal (pd.DataFrame): DataFrame containing signal data with columns "time" and "value".
    - window_size (int): The size of the rolling window.
    - threshold (float): The z-score threshold to determine anomalies.

    Returns:
    - List of tuples with (start_time, end_time) indicating detected anomaly periods.
    """

    # Calculate mean and std of background noise
    noise_mean = background_noise["value"].mean()
    noise_std = background_noise["value"].std()

    # Initialize list to store anomaly periods
    anomalies = []
    current_anomaly_start = None

    # Rolling window through signal data
    for i in range(len(signal) - window_size + 1):
        # Get the window of data
        window = signal["value"].iloc[i : i + window_size]

        # Calculate z-score based on background noise stats
        window_mean = window.mean()
        z_score = (window_mean - noise_mean) / noise_std

        # Check if the z-score indicates an anomaly
        if abs(z_score) > threshold:
            # If this is the start of an anomaly period, record it
            if current_anomaly_start is None:
                current_anomaly_start = i
        else:
            # If an anomaly period ends, finalize the current anomaly
            if current_anomaly_start is not None:
                start_time = signal["time"].iloc[current_anomaly_start]
                end_time = signal["time"].iloc[i + window_size - 1]
                anomalies.append((start_time, end_time))
                current_anomaly_start = None

    # If an anomaly is ongoing at the end, close it
    if current_anomaly_start is not None:
        start_time = signal["time"].iloc[current_anomaly_start]
        end_time = signal["time"].iloc[len(signal) - 1]
        anomalies.append((start_time, end_time))

    # Merge overlapping or adjacent anomaly periods
    merged_anomalies = []
    for start, end in anomalies:
        if merged_anomalies and merged_anomalies[-1][1] >= start:
            merged_anomalies[-1] = (merged_anomalies[-1][0], max(merged_anomalies[-1][1], end))
        else:
            merged_anomalies.append((start, end))

    return merged_anomalies
