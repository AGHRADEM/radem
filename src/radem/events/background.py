import pandas as pd
import numpy as np
from scipy.stats import zscore

def intervals_from_mask(mask, signal):
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


def autosplit_poisson_mask(background_noise, signal, window_size, threshold):
    p_lambda = background_noise["value"].mean()

    mask = pd.Series(False, index=signal.index)

    for i in range(len(signal) - window_size + 1):
        window = signal["value"].iloc[i : i + window_size]

        window_mean = window.mean()
        z_score = (window_mean - p_lambda) / np.sqrt(p_lambda / window_size)

        if abs(z_score) > threshold:
            mask.iloc[i : i + window_size] = True

    return mask


def autosplit_poisson(background_noise, signal, window_size=2*24*60, threshold=2):
    mask = autosplit_poisson_mask(background_noise, signal, window_size, threshold)
    return intervals_from_mask(mask, signal)