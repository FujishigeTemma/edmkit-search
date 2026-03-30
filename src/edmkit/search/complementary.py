from collections.abc import Callable, Iterator

import numpy as np
from edmkit.metrics import MetricFunc
from edmkit.splits import SplitFunc
from edmkit.types import PredictFunc

from .common import score_subset_per_fold
from .dataset import Dataset, Subset
from .types import FilterFn, Step

WeightFunc = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
"""Fold weight function: (fold_scores, prev_fold_scores, prev_weights) -> new_weights.

Called after each variable is selected to update fold weights for the next step.
Must return a non-negative array that sums to 1.
"""


def softmax_weight(*, temperature: float = 1.0) -> WeightFunc:
    """Create a softmax fold-weighting strategy.

    Recomputes weights from current fold scores each step.  Low-scoring
    folds receive higher weight, encouraging the next variable to help
    where prediction is weakest.

    Parameters
    ----------
    temperature : float
        Controls the balance between generalist and specialist selection.
        Large values produce near-uniform weights (generalist); values
        near zero concentrate weight on the worst fold (specialist).

    Returns
    -------
    WeightFunc
        ``(fold_scores, prev_fold_scores, prev_weights) -> new_weights``.

    Raises
    ------
    ValueError
        If temperature is not positive.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    def fn(
        fold_scores: np.ndarray, prev_fold_scores: np.ndarray, prev_weights: np.ndarray
    ) -> np.ndarray:
        logits = -fold_scores / temperature
        logits = logits - logits.max()
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum()

    return fn


def greedy_complementary(
    data: Dataset | Subset,
    *,
    predict: PredictFunc,
    metric: MetricFunc,
    split: SplitFunc,
    weight: WeightFunc | None = None,
    threshold: float = 0.0,
    max_dim: int = 10,
    filter: FilterFn | None = None,
) -> Iterator[Step]:
    """Greedily select variables using fold-weighted complementarity.

    Unlike :func:`greedy`, this function evaluates candidates across
    multiple temporal folds and weights fold-level improvements so that
    poorly-predicted folds receive more attention.  This balances
    *generalist* variables (good everywhere) with *specialist* variables
    (excellent in specific temporal regimes).

    Parameters
    ----------
    data : Dataset | Subset
        Full dataset.  Split into folds internally via *split*.
    predict : PredictFunc
        Prediction function ``(X, Y, Q, *, mask) -> predictions``.
    metric : MetricFunc
        Metric function for evaluating prediction quality.
    split : SplitFunc
        Splitting strategy ``(n,) -> list[Fold]``.
    weight : WeightFunc | None
        Fold weight update function.  Default is ``softmax_weight()``.
    threshold : float
        Minimum mean fold score for candidate selection.  Default is 0.0.
    max_dim : int
        Maximum number of variables to select.  Default is 10.
    filter : FilterFn | None
        Optional filter ``(x, Y) -> bool`` to accept/reject candidates.

    Yields
    ------
    Step
        Result of each dimension selection step.  ``Step.score`` is the
        mean across folds (not the weighted score) for interpretability.
    """
    X = data.X
    Y = data.Y
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got {X.ndim}D with shape {X.shape}")
    if Y.ndim == 1:
        Y = Y[:, None]

    N = X.shape[1]
    if max_dim > N:
        raise ValueError(f"max_dim must be <= N (={N}), got {max_dim}")

    folds = split(X.shape[0])
    if len(folds) == 0:
        raise ValueError("split produced no folds")

    K = len(folds)
    weight_func = weight if weight is not None else softmax_weight()

    available: set[int] = set(range(N))
    selected_indices: list[int] = []
    current_fold_scores = np.zeros(K)
    weights = np.full(K, 1.0 / K)

    for dim in range(1, max_dim + 1):
        results = []
        for v in available:
            fold_scores = score_subset_per_fold(
                selected_indices + [v],
                folds=folds,
                X=X,
                Y=Y,
                predict=predict,
                metric=metric,
            )
            delta = fold_scores - current_fold_scores
            weighted_improvement = float(np.dot(weights, delta))
            mean_score = float(np.mean(fold_scores))
            results.append((v, weighted_improvement, mean_score, fold_scores))

        candidates = sorted(
            ((v, w, ms, fs) for v, w, ms, fs in results if ms >= threshold),
            key=lambda r: r[1],
            reverse=True,
        )

        best = None
        for v, w, mean_score, fold_scores in candidates:
            if filter is not None and not filter(X[:, v], Y):
                continue
            best = (v, mean_score, fold_scores)
            break

        if best is None:
            return

        best_v, best_mean_score, best_fold_scores = best
        weights = weight_func(best_fold_scores, current_fold_scores, weights)
        current_fold_scores = best_fold_scores
        selected_indices.append(best_v)
        available.remove(best_v)

        yield Step(
            index=best_v,
            score=best_mean_score,
            selected=tuple(selected_indices),
        )
