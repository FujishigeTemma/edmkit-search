# ruff: noqa: F401
"""Multi-target MDE (Manifold Dimension Expansion) for causal discovery."""

from .mde import ccm_convergence_test, evaluate_manifold, mde
from .metrics import MetricFn, mae, mean_rho, rmse
from .types import DataSplit, MDEResult
from .validation import split_data
