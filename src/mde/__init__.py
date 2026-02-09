# ruff: noqa: F401
"""MDE (Manifold Dimension Expansion) for causal discovery."""

# High-level API
from .ccm import ccm_converged, ccm_convergence_diagnostics, CCMDiagnostics, make_ccm_filter
from .selection import (
    mde,
    select,
    select_iter,
    get_predictions,
    evaluate_manifold,
    MDEResult,
    SelectionStep,
    SelectionResult,
    EvaluationResult,
    MetricConfig,
    CandidateFilter,
)
from .splits import temporal_split, TemporalSplit

# Building blocks
from .skill import prediction_skill, PredictFn

# Metrics
from .metrics import MetricFn, mae, mean_rho, negate, rmse

__all__ = [
    # High-level API
    "mde",
    "select",
    "select_iter",
    "get_predictions",
    "evaluate_manifold",
    "ccm_converged",
    "ccm_convergence_diagnostics",
    "make_ccm_filter",
    "temporal_split",
    # Building blocks
    "prediction_skill",
    # Types
    "PredictFn",
    "MetricFn",
    "CandidateFilter",
    "TemporalSplit",
    "MDEResult",
    "SelectionStep",
    "SelectionResult",
    "EvaluationResult",
    "CCMDiagnostics",
    "MetricConfig",
    # Metrics
    "mae",
    "rmse",
    "mean_rho",
    "negate",
]
