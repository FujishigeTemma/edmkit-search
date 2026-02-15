# ruff: noqa: F401
"""MDE (Manifold Dimension Expansion) for causal discovery."""

# Core algorithm
from .core import mde, build_result, get_predictions, evaluate_manifold, evaluate_expanding

# Search
from .search import greedy, greedy_iter

# Building blocks
from .skill import prediction_skill

# Dataset
from .dataset import Dataset, Subset, temporal_split, make_expanding_windows, make_sliding_windows

# Metrics
from .metrics import mae, mean_rho, negate, rmse
from .metrics import mae_per_dim, mean_rho_per_dim, rmse_per_dim

# Types
from .core import Result, Evaluation
from .search import Selection, Filter, Step
from .dataset import Fold, Transform
from .skill import PredictFn
from .metrics import MetricFn, PerDimMetricFn
