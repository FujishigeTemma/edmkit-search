from collections.abc import Callable
from typing import TypeAlias

import numpy as np

PredictFn: TypeAlias = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
"""(X_train, Y_train, X_query) -> predictions"""

MetricFn: TypeAlias = Callable[[np.ndarray, np.ndarray], float]
"""(predictions, observations) -> score"""

FilterFn: TypeAlias = Callable[[np.ndarray, np.ndarray], bool]
"""(x, Y) -> accept"""
