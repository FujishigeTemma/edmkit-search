from collections.abc import Callable, Iterator

import numpy as np
from edmkit.metrics import MetricFunc
from edmkit.splits import Fold, SplitFunc
from edmkit.types import PredictFunc

from .dataset import Dataset, Subset
from .types import FilterFn, SampleLossFn, Step

WeightFunc = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
"""Weight update function: (values, prev_values, prev_weights) -> new_weights."""


def validate_temperature(temperature: float) -> None:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")


def softmax(values: np.ndarray, *, temperature: float, maximize: bool) -> np.ndarray:
    direction = -1.0 if maximize else 1.0
    logits = direction * values / temperature
    logits = logits - logits.max()
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum()


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
    validate_temperature(temperature)

    def fn(
        scores: np.ndarray, prev_scores: np.ndarray, prev_weights: np.ndarray
    ) -> np.ndarray:
        del prev_scores, prev_weights
        return softmax(scores, temperature=temperature, maximize=True)

    return fn


def softmax_loss_weight(*, temperature: float = 1.0) -> WeightFunc:
    """Create a softmax value-weighting strategy.

    Higher lower-is-better objective values receive higher weight,
    encouraging the next variable to focus on poorly explained timepoints.

    Parameters
    ----------
    temperature : float
        Controls how concentrated the weighting is. Large values produce
        near-uniform weights; values near zero concentrate on the highest
        losses.

    Returns
    -------
    WeightFunc
        ``(values, prev_values, prev_weights) -> new_weights``.

    Raises
    ------
    ValueError
        If temperature is not positive.
    """
    validate_temperature(temperature)

    def fn(
        losses: np.ndarray, prev_losses: np.ndarray, prev_weights: np.ndarray
    ) -> np.ndarray:
        del prev_losses, prev_weights
        return softmax(losses, temperature=temperature, maximize=False)

    return fn


def _as_matching_2d_arrays(
    predictions: np.ndarray,
    observations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if predictions.shape != observations.shape:
        raise ValueError(
            f"predictions and observations must have the same shape, got "
            f"{predictions.shape} and {observations.shape}"
        )
    if predictions.ndim == 1:
        return predictions[:, None], observations[:, None]
    if predictions.ndim != 2:
        raise ValueError(
            f"predictions and observations must be 1D or 2D, got {predictions.ndim}D"
        )
    return predictions, observations


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
    predictions, observations = _as_matching_2d_arrays(predictions, observations)
    return np.abs(predictions - observations).mean(axis=1)


def mean_squared_error_per_sample(
    predictions: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    """Compute mean squared error for each sample.

    Parameters
    ----------
    predictions : np.ndarray of shape (N,) or (N, D)
        Predicted values.
    observations : np.ndarray of shape (N,) or (N, D)
        Observed values with the same shape as *predictions*.

    Returns
    -------
    np.ndarray of shape (N,)
        Mean squared error for each sample, averaged across dimensions.

    Raises
    ------
    ValueError
        If the input shapes do not match or are not 1D/2D.
    """
    predictions, observations = _as_matching_2d_arrays(predictions, observations)
    return ((predictions - observations) ** 2).mean(axis=1)


def mean_negative_correlation_contribution_per_sample(
    predictions: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    """Compute a per-sample objective aligned with Pearson correlation.

    Each sample receives the negative of its standardized covariance
    contribution, averaged across dimensions. Lower values are better.
    Summing these contributions over samples recovers the numerator term
    of Pearson correlation after normalization, so weighting them is
    comparable in intent to weighting fold-level correlation scores.

    Parameters
    ----------
    predictions : np.ndarray of shape (N,) or (N, D)
        Predicted values.
    observations : np.ndarray of shape (N,) or (N, D)
        Observed values with the same shape as *predictions*.

    Returns
    -------
    np.ndarray of shape (N,)
        Negative correlation contribution for each sample, averaged across
        dimensions.

    Raises
    ------
    ValueError
        If the input shapes do not match or are not 1D/2D.
    """
    predictions, observations = _as_matching_2d_arrays(predictions, observations)

    p_centered = predictions - predictions.mean(axis=0, keepdims=True)
    o_centered = observations - observations.mean(axis=0, keepdims=True)
    denom = np.sqrt((p_centered**2).sum(axis=0) * (o_centered**2).sum(axis=0))
    safe_denom = np.where(denom > 0, denom, 1.0)
    contributions = (p_centered * o_centered) / safe_denom
    contributions = np.where(denom > 0, contributions, 0.0)
    return -contributions.mean(axis=1)


def _validate_data(
    data: Dataset | Subset, *, max_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    X = data.X
    Y = data.Y

    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got {X.ndim}D with shape {X.shape}")
    if Y.ndim == 1:
        Y = Y[:, None]
    elif Y.ndim != 2:
        raise ValueError(
            f"Y must be 1D or 2D array, got {Y.ndim}D with shape {Y.shape}"
        )

    M = X.shape[1]
    if max_dim > M:
        raise ValueError(f"max_dim must be <= M (={M}), got {max_dim}")

    return X, Y


def _validate_folds(folds: list[Fold]) -> None:
    if len(folds) == 0:
        raise ValueError("split produced no folds")


def _select_candidate(
    candidates: list[tuple[int, float, np.ndarray]],
    *,
    X: np.ndarray,
    Y: np.ndarray,
    filter: FilterFn | None,
) -> tuple[int, float, np.ndarray] | None:
    for index, score, state in candidates:
        if filter is not None and not filter(X[:, index], Y):
            continue
        return index, score, state
    return None


def _run_complementary_greedy(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    max_dim: int,
    initial_state: np.ndarray,
    initial_weights: np.ndarray,
    evaluate_candidates: Callable[
        [list[int], set[int], np.ndarray, np.ndarray],
        list[tuple[int, float, float, np.ndarray]],
    ],
    update_weights: WeightFunc,
    filter: FilterFn | None,
    threshold: float,
) -> Iterator[Step]:
    available: set[int] = set(range(X.shape[1]))
    selected_indices: list[int] = []
    current_state = initial_state
    weights = initial_weights

    for _ in range(max_dim):
        results = evaluate_candidates(
            selected_indices, available, current_state, weights
        )
        candidates = sorted(
            (
                (index, improvement, mean_score, state)
                for index, improvement, mean_score, state in results
                if mean_score >= threshold
            ),
            key=lambda candidate: candidate[1],
            reverse=True,
        )
        best = _select_candidate(
            [(index, mean_score, state) for index, _, mean_score, state in candidates],
            X=X,
            Y=Y,
            filter=filter,
        )
        if best is None:
            return

        best_index, best_score, best_state = best
        weights = update_weights(best_state, current_state, weights)
        current_state = best_state
        selected_indices.append(best_index)
        available.remove(best_index)

        yield Step(
            index=best_index,
            score=best_score,
            selected=tuple(selected_indices),
        )


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

    def evaluate_candidates(
        selected_indices: list[int],
        available: set[int],
        current_scores: np.ndarray,
        weights: np.ndarray,
    ) -> list[tuple[int, float, float, np.ndarray]]:
        results = []
        for index in available:
            fold_scores = _fold_scores(
                selected_indices + [index],
                folds=folds,
                X=X,
                Y=Y,
                predict=predict,
                metric=metric,
            )
            improvement = float(np.dot(weights, fold_scores - current_scores))
            mean_score = float(np.mean(fold_scores))
            results.append((index, improvement, mean_score, fold_scores))
        return results

    yield from _run_complementary_greedy(
        X=X,
        Y=Y,
        max_dim=max_dim,
        initial_state=np.zeros(K),
        initial_weights=np.full(K, 1.0 / K),
        evaluate_candidates=evaluate_candidates,
        update_weights=weight_func,
        filter=filter,
        threshold=threshold,
    )


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
    used for the reported score and thresholding, while ``loss``
    provides a metric-aligned lower-is-better per-sample objective that
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
        Per-sample lower-is-better objective returning ``(N_validation,)``
        for each fold. Default is ``mean_abs_error_per_sample``.
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

    def evaluate_candidates(
        selected_indices: list[int],
        available: set[int],
        current_losses: np.ndarray,
        weights: np.ndarray,
    ) -> list[tuple[int, float, float, np.ndarray]]:
        results = []
        for index in available:
            fold_scores, sample_losses = _fold_scores_and_losses(
                selected_indices + [index],
                folds=folds,
                X=X,
                Y=Y,
                predict=predict,
                metric=metric,
                loss=loss,
            )
            improvement = float(np.dot(weights, current_losses - sample_losses))
            mean_score = float(np.mean(fold_scores))
            results.append((index, improvement, mean_score, sample_losses))
        return results

    yield from _run_complementary_greedy(
        X=X,
        Y=Y,
        max_dim=max_dim,
        initial_state=np.zeros(sample_count),
        initial_weights=np.full(sample_count, 1.0 / sample_count),
        evaluate_candidates=evaluate_candidates,
        update_weights=weight_func,
        filter=filter,
        threshold=threshold,
    )
