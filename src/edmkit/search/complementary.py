from collections.abc import Callable, Iterator

import numpy as np
from edmkit.metrics import MetricFunc
from edmkit.splits import Fold, SplitFunc
from edmkit.types import PredictFunc

from .dataset import Dataset, Subset
from .types import FilterFn, SampleLossFn, Step

WeightFunc = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
"""Weight update function: (values, prev_values, prev_weights) -> new_weights."""


def softmax_weight(*, temperature: float = 1.0) -> WeightFunc:
    """Create a softmax score-weighting strategy.

    Lower scores receive higher weight, encouraging the next variable to
    improve weak folds.

    Parameters
    ----------
    temperature : float
        Controls how concentrated the weighting is. Large values produce
        near-uniform weights; values near zero concentrate on the lowest
        scores.

    Returns
    -------
    WeightFunc
        ``(scores, prev_scores, prev_weights) -> new_weights``.

    Raises
    ------
    ValueError
        If temperature is not positive.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    def fn(
        scores: np.ndarray, prev_scores: np.ndarray, prev_weights: np.ndarray
    ) -> np.ndarray:
        del prev_scores, prev_weights
        logits = -scores / temperature
        logits = logits - logits.max()
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum()

    return fn


def softmax_loss_weight(*, temperature: float = 1.0) -> WeightFunc:
    """Create a softmax loss-weighting strategy.

    Higher losses receive higher weight, encouraging the next variable to
    focus on poorly predicted timepoints.

    Parameters
    ----------
    temperature : float
        Controls how concentrated the weighting is. Large values produce
        near-uniform weights; values near zero concentrate on the highest
        losses.

    Returns
    -------
    WeightFunc
        ``(losses, prev_losses, prev_weights) -> new_weights``.

    Raises
    ------
    ValueError
        If temperature is not positive.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    def fn(
        losses: np.ndarray, prev_losses: np.ndarray, prev_weights: np.ndarray
    ) -> np.ndarray:
        del prev_losses, prev_weights
        logits = losses / temperature
        logits = logits - logits.max()
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum()

    return fn


def mean_abs_error_per_sample(
    predictions: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    """Compute mean absolute error for each sample.

    Parameters
    ----------
    predictions : np.ndarray of shape (N,) or (N, D)
        Predicted values.
    observations : np.ndarray of shape (N,) or (N, D)
        Observed values with the same shape as *predictions*.

    Returns
    -------
    np.ndarray of shape (N,)
        Mean absolute error for each sample, averaged across dimensions.

    Raises
    ------
    ValueError
        If the input shapes do not match or are not 1D/2D.
    """
    if predictions.shape != observations.shape:
        raise ValueError(
            f"predictions and observations must have the same shape, got "
            f"{predictions.shape} and {observations.shape}"
        )
    if predictions.ndim == 1:
        predictions = predictions[:, None]
        observations = observations[:, None]
    elif predictions.ndim != 2:
        raise ValueError(
            f"predictions and observations must be 1D or 2D, got {predictions.ndim}D"
        )
    return np.abs(predictions - observations).mean(axis=1)


def _validate_data(data: Dataset | Subset, *, max_dim: int) -> tuple[np.ndarray, np.ndarray]:
    X = data.X
    Y = data.Y

    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got {X.ndim}D with shape {X.shape}")
    if Y.ndim == 1:
        Y = Y[:, None]
    elif Y.ndim != 2:
        raise ValueError(f"Y must be 1D or 2D array, got {Y.ndim}D with shape {Y.shape}")

    M = X.shape[1]
    if max_dim > M:
        raise ValueError(f"max_dim must be <= M (={M}), got {max_dim}")

    return X, Y


def _validate_folds(folds: list[Fold]) -> None:
    if len(folds) == 0:
        raise ValueError("split produced no folds")


def _predict_subset(
    indices: list[int],
    *,
    X_train: np.ndarray,
    X_validation: np.ndarray,
    Y_train: np.ndarray,
    predict: PredictFunc,
) -> np.ndarray:
    predictions = predict(X_train[:, indices], Y_train, X_validation[:, indices])
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    return predictions


def _fold_scores(
    indices: list[int],
    *,
    folds: list[Fold],
    X: np.ndarray,
    Y: np.ndarray,
    predict: PredictFunc,
    metric: MetricFunc,
) -> np.ndarray:
    scores = np.empty(len(folds))
    for k, fold in enumerate(folds):
        predictions = _predict_subset(
            indices,
            X_train=X[fold.train],
            X_validation=X[fold.validation],
            Y_train=Y[fold.train],
            predict=predict,
        )
        scores[k] = float(metric(predictions, Y[fold.validation]))
    return scores


def _fold_scores_and_losses(
    indices: list[int],
    *,
    folds: list[Fold],
    X: np.ndarray,
    Y: np.ndarray,
    predict: PredictFunc,
    metric: MetricFunc,
    loss: SampleLossFn,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.empty(len(folds))
    losses = []

    for k, fold in enumerate(folds):
        predictions = _predict_subset(
            indices,
            X_train=X[fold.train],
            X_validation=X[fold.validation],
            Y_train=Y[fold.train],
            predict=predict,
        )
        observations = Y[fold.validation]
        scores[k] = float(metric(predictions, observations))
        losses.append(loss(predictions, observations))

    return scores, np.concatenate(losses)


def greedy_complementary_folds(
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

    Candidates are evaluated on each fold independently, then compared by
    the weighted improvement over the current fold scores. ``Step.score``
    is the mean fold score for interpretability.

    Parameters
    ----------
    data : Dataset | Subset
        Full dataset. Split into folds internally via *split*.
    predict : PredictFunc
        Prediction function ``(X, Y, Q, *, mask) -> predictions``.
    metric : MetricFunc
        Fold-level evaluation metric. Higher values must be better.
    split : SplitFunc
        Splitting strategy ``(n,) -> list[Fold]``.
    weight : WeightFunc | None
        Fold weight update function. Default is ``softmax_weight()``.
    threshold : float
        Minimum mean fold score for candidate selection. Default is 0.0.
    max_dim : int
        Maximum number of variables to select. Default is 10.
    filter : FilterFn | None
        Optional filter ``(x, Y) -> bool`` to accept/reject candidates.

    Yields
    ------
    Step
        Result of each dimension selection step.

    Raises
    ------
    ValueError
        If the inputs are invalid or *split* produces no folds.
    """
    X, Y = _validate_data(data, max_dim=max_dim)
    folds = split(X.shape[0])
    _validate_folds(folds)

    K = len(folds)
    weight_func = weight if weight is not None else softmax_weight()

    available: set[int] = set(range(X.shape[1]))
    selected_indices: list[int] = []
    current_scores = np.zeros(K)
    weights = np.full(K, 1.0 / K)

    for _ in range(max_dim):
        results = []
        for v in available:
            fold_scores = _fold_scores(
                selected_indices + [v],
                folds=folds,
                X=X,
                Y=Y,
                predict=predict,
                metric=metric,
            )
            delta = fold_scores - current_scores
            weighted_improvement = float(np.dot(weights, delta))
            mean_score = float(np.mean(fold_scores))
            results.append((v, weighted_improvement, mean_score, fold_scores))

        candidates = sorted(
            ((v, improvement, mean_score, scores)
             for v, improvement, mean_score, scores in results
             if mean_score >= threshold),
            key=lambda result: result[1],
            reverse=True,
        )

        best = None
        for v, _, mean_score, fold_scores in candidates:
            if filter is not None and not filter(X[:, v], Y):
                continue
            best = (v, mean_score, fold_scores)
            break

        if best is None:
            return

        best_v, best_mean_score, best_fold_scores = best
        weights = weight_func(best_fold_scores, current_scores, weights)
        current_scores = best_fold_scores
        selected_indices.append(best_v)
        available.remove(best_v)

        yield Step(index=best_v, score=best_mean_score, selected=tuple(selected_indices))


def greedy_complementary_timepoints(
    data: Dataset | Subset,
    *,
    predict: PredictFunc,
    metric: MetricFunc,
    split: SplitFunc,
    loss: SampleLossFn = mean_abs_error_per_sample,
    weight: WeightFunc | None = None,
    threshold: float = 0.0,
    max_dim: int = 10,
    filter: FilterFn | None = None,
) -> Iterator[Step]:
    """Greedily select variables using timepoint-weighted complementarity.

    Candidates are evaluated on each fold, but the complementary weighting
    is applied to the concatenated validation timepoints. ``metric`` is
    used only for the reported score and thresholding, while ``loss``
    determines which timepoints receive more attention.

    Parameters
    ----------
    data : Dataset | Subset
        Full dataset. Split into folds internally via *split*.
    predict : PredictFunc
        Prediction function ``(X, Y, Q, *, mask) -> predictions``.
    metric : MetricFunc
        Fold-level evaluation metric. Higher values must be better.
    split : SplitFunc
        Splitting strategy ``(n,) -> list[Fold]``.
    loss : SampleLossFn
        Per-sample loss function returning ``(N_validation,)`` for each fold.
        Default is ``mean_abs_error_per_sample``.
    weight : WeightFunc | None
        Timepoint weight update function. Default is ``softmax_loss_weight()``.
    threshold : float
        Minimum mean fold score for candidate selection. Default is 0.0.
    max_dim : int
        Maximum number of variables to select. Default is 10.
    filter : FilterFn | None
        Optional filter ``(x, Y) -> bool`` to accept/reject candidates.

    Yields
    ------
    Step
        Result of each dimension selection step.

    Raises
    ------
    ValueError
        If the inputs are invalid or *split* produces no folds.
    """
    X, Y = _validate_data(data, max_dim=max_dim)
    folds = split(X.shape[0])
    _validate_folds(folds)

    sample_count = sum(len(fold.validation) for fold in folds)
    weight_func = weight if weight is not None else softmax_loss_weight()

    available: set[int] = set(range(X.shape[1]))
    selected_indices: list[int] = []
    current_losses = np.zeros(sample_count)
    weights = np.full(sample_count, 1.0 / sample_count)

    for _ in range(max_dim):
        results = []
        for v in available:
            fold_scores, sample_losses = _fold_scores_and_losses(
                selected_indices + [v],
                folds=folds,
                X=X,
                Y=Y,
                predict=predict,
                metric=metric,
                loss=loss,
            )
            improvement = current_losses - sample_losses
            weighted_improvement = float(np.dot(weights, improvement))
            mean_score = float(np.mean(fold_scores))
            results.append((v, weighted_improvement, mean_score, sample_losses))

        candidates = sorted(
            ((v, improvement, mean_score, sample_losses)
             for v, improvement, mean_score, sample_losses in results
             if mean_score >= threshold),
            key=lambda result: result[1],
            reverse=True,
        )

        best = None
        for v, _, mean_score, sample_losses in candidates:
            if filter is not None and not filter(X[:, v], Y):
                continue
            best = (v, mean_score, sample_losses)
            break

        if best is None:
            return

        best_v, best_mean_score, best_sample_losses = best
        weights = weight_func(best_sample_losses, current_losses, weights)
        current_losses = best_sample_losses
        selected_indices.append(best_v)
        available.remove(best_v)

        yield Step(index=best_v, score=best_mean_score, selected=tuple(selected_indices))

