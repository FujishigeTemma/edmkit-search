# ruff: noqa: F401
"""MDE: greedy variable selection for Empirical Dynamic Modeling."""

# Search (core algorithm)
from .search import greedy, greedy_iter

# Convergence testing
from .convergence import causation, causation_iter

# Dataset
from .dataset import (
    Dataset,
    Subset,
    temporal_split,
    expanding_splits,
    sliding_splits,
)

# Metrics
from .metrics import mae, mean_rho, negate, rmse

# Types (callback interfaces)
from .types import FilterFn, MetricFn, PredictFn

# Types (data structures)
from .dataset import Fold, Transform
from .metrics import MetricConfig
from .search import Selection, Step
