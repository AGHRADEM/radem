import pandas as pd
import numpy as np
import datetime 
from typing import List, Tuple

def find_events(series: pd.Series, window: int, critical_value: float = 2.0) -> List[Tuple[pd.Index, pd.Index]]:
    """
    Identify events in a time series where the values exceed a defined threshold
    based on a rolling mean and standard deviation.

    Parameters:
    ----------
    series : pd.Series
        A pandas Series containing the time series data.
    window : int
        The size of the rolling window used to calculate the moving average and standard deviation.
    critical_value : float, optional
        A multiplier for the standard deviation to define the threshold for events.

    Returns:
    -------
    List[Tuple[pd.Index, pd.Index]]
        A list of tuples, where each tuple contains the start and end index of an identified event.
    """

    moving_avg = series.rolling(window=window).mean()
    moving_std = series.rolling(window=window).std()

    is_event = (series > (moving_avg + critical_value * moving_std))

    # Identify events
    events = []
    in_event = False
    event_start = None

    for i in range(len(series)):
        if is_event.iloc[i]:
            if not in_event:
                # Mark the start of the spike
                event_start = series.index[i]
                in_event = True
        else:
            if in_event:
                  # Mark the end of the spike
                event_end = series.index[i - 1]
                events.append((event_start, event_end))
                in_event = False

    # If the last point is a spike, close the event
    if in_event:
        events.append((event_start, series.index[-1]))

    return events


def intervals_from_mask(mask: np.array) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    # Find the change points in the mask
    changes = np.diff(mask.astype(int))

    # Start of intervals where mask is True (change from 0 to 1)
    starts = np.where(changes == 1)[0] + 1

    # End of intervals where mask is True (change from 1 to 0)
    ends = np.where(changes == -1)[0] + 1

    # Handle edge cases where the mask starts or ends with True
    if mask[0]:
        starts = np.insert(starts, 0, 0)
    if mask[-1]:
        ends = np.append(ends, len(mask))

    intervals = list(zip(starts, ends))
    return intervals

def noise_mask(df: pd.DataFrame, rolling_window: int, offsets: List[int], full_interval: Tuple[datetime.datetime, datetime.datetime] = None) -> np.array:
    """Returns only the noise from the full search interval"""
    if not full_interval is None:
        start, end = full_interval
        df = df[(df['time'] >= start) & (df['time'] <= end)]

    rolling_mean = np.array(df["value"].rolling(window=rolling_window, center=True).mean())
    rolling_std = np.array(df["value"].rolling(window=rolling_window, center=True).std())

    mask = np.ones(len(df), dtype=bool) # Accepts all
    for offset in offsets:
        z_scores_forward = np.zeros(len(rolling_mean))
        z_scores_backward = np.zeros(len(rolling_mean))
        for i in range(offset, len(df) - offset):
            z_scores_forward[i] = (rolling_mean[i] - rolling_mean[i + offset]) / rolling_std[i + offset]
            z_scores_backward[i] = (rolling_mean[i] - rolling_mean[i - offset]) / rolling_std[i - offset]
        noise_mask = (np.abs(z_scores_forward) < 3) & (np.abs(z_scores_backward) < 3)
        mask = mask & noise_mask
    
    
    return mask