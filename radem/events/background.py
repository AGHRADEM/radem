import pandas as pd
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