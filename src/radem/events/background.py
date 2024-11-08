import pandas as pd
import numpy as np
from scipy.stats import zscore

def autosplit_gauss(background_noise, signal, window_size=50, threshold=3):
    """
    Detects anomalies in signal data based on rolling window z-score comparison with background noise.

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
    is_anomaly = False
    start_index = None

    # Rolling window through signal data
    for i in range(len(signal) - window_size + 1):
        # Get the window of data
        window = signal["value"].iloc[i : i + window_size]

        # Calculate z-score based on background noise stats
        window_mean = window.mean()
        z_score = (window_mean - noise_mean) / noise_std

        # Check if the z-score indicates an anomaly
        if abs(z_score) > threshold:
            if not is_anomaly:  # Start of a new anomaly period
                is_anomaly = True
                start_index = i
        else:
            if is_anomaly:  # End of an anomaly period
                start_time = signal["time"].iloc[start_index]
                end_time = signal["time"].iloc[i + window_size - 1]
                anomalies.append((start_time, end_time))
                is_anomaly = False

    # If an anomaly is ongoing at the end, close it
    if is_anomaly:
        start_time = signal["time"].iloc[start_index]
        end_time = signal["time"].iloc[len(signal) - 1]
        anomalies.append((start_time, end_time))

    return anomalies
