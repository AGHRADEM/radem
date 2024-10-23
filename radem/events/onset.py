import pandas as pd
import numpy as np

def detect_onset(series: pd.Series, mean: float, sigma: float, window: int = 30, critical_value: float = 2) -> pd.Timestamp | None:
    """
    Detects the onset time of an anomaly in a time series based on a CUSUM-like algorithm. Can be used for solar energetic particle detection.

    Parameters:
    -----------
    series: A pandas Series representing the time series data with a datetime index.
    mean: The expected mean value of the background distribution (e.g., background radiation).
    sigma: The standard deviation of the background distribution.
    window: The number of consecutive time points required to confirm an onset event.
    critical_value: The threshold multiplier applied to sigma to define the uncertainty limit.

    Returns:
    --------
    pd.Timestamp: The timestamp of the detected onset time, or None if no onset is detected.

    Raises:
    -------
    ValueError: If the mean value and uncertainty limit are equal, making it impossible to compute the control parameter.

    Notes:
    ------
    The function uses a control parameter `k` based on the difference between the uncertainty limit and the mean.
    A cumulative sum (CUSUM) approach is used to detect periods where the signal consistently exceeds the threshold.
    """
    
    uncertainty_limit = mean + critical_value * sigma

    # Prevent division by zero or invalid log calculations.
    if mean == uncertainty_limit:
        raise ValueError("""Background radiation mean value and uncertainty limit are equal.
                            Control parameter k cannot be computed.""")

    # Compute the control parameter 'k' used to scale the CUSUM process.
    k = (uncertainty_limit - mean) / (np.log1p(uncertainty_limit) - np.log1p(mean))
    hastiness = 1 if k < 1.0 else 2  # Determine the threshold for CUSUM alert triggering.

    alert = 0  # Tracks consecutive exceedances of the threshold.
    previous_cusum, cusum = 0, 0

    onset_time = None

    for i in range(1, len(series)):
        # Normalize the current flux value based on the mean and sigma.
        normalized_flux = (series.iloc[i] - mean) / sigma
        previous_cusum, cusum = cusum, max(0, normalized_flux - round(k) + previous_cusum)
        
        # Increment alert counter if CUSUM exceeds the hastiness threshold.
        # If alert reaches the required window size, register the onset time.
        alert = alert + 1 if cusum > hastiness else 0
        if alert == window:
            onset_time = series.index[i - alert]
            break
    
    return onset_time
