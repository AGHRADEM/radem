import pandas as pd
import numpy as np
import scipy

from dataclasses import dataclass
from typing import List
from datetime import datetime

@dataclass
class EventInterval:
    start: datetime
    stop: datetime

def autosplit_events(search_series: pd.Series, background_sample: pd.Series, window: int, method: str = "gaussian") -> List[EventInterval]:
    t_test = search_series.rolling(window=window).apply(lambda w: scipy.stats.ttest_ind(w, background_sample))
    print(t_test)