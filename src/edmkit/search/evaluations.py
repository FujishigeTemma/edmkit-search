"""Candidate-evaluation factories.

Each factory captures the data / predict / metric / split / weighting in
a closure and returns a higher-order :class:`~edmkit.search.types.Evaluation`
function. Call the returned evaluation with a strategy to run a search:

    evaluation = holdout(
        X_train=..., X_val=..., Y_train=..., Y_val=...,
        predict=simplex_projection, metric=mean_rho,
    )
    steps = list(evaluation(greedy, max_dim=10))

Five built-in evaluations cover the (split, weighting) cross-product:

* :func:`holdout`              — single train/val split, state = ``None``
* :func:`folds`                — k-fold aggregate, state = ``None``
* :func:`loo`                  — leave-one-out + Theiler exclusion, state = ``None``
* :func:`weighted_folds`       — k-fold with per-fold weighting,
  state = per-fold scores ``(K,)``
* :func:`weighted_timepoints`  — k-fold with per-sample weighting,
  state = per-sample losses ``(N_total,)``

Per-sample loss helpers (``mean_abs_error_per_sample`` etc.) and weight
constructors (``softmax_weight``, ``softmax_loss_weight``) live here as
the building blocks of the adaptive evaluations.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Protocol

import numpy as np
from edmkit.metrics import MetricFunc
from edmkit.simplex_projection import loo as simplex_loo
from edmkit.splits import Fold
from edmkit.types import PredictFunc

from . import FilterFn, Step, Strategy


class SampleLossFn(Protocol):
    """``(predictions, observations) -> per-sample lower-is-better loss``."""

    __name__: str

    def __call__(
        self,
        predictions: np.ndarray,
        observations: np.ndarray,
        /,
    ) -> np.ndarray: ...


class WeightFunc(Protocol):
    """``(state) -> weights``.

    For ``weighted_folds`` the input is parent per-fold scores ``(K,)``.
    For ``weighted_timepoints`` it is parent per-sample losses
    ``(N_total,)``. Returned weights share the input shape and sum to one.
    """

    def __call__(self, state: np.ndarray) -> np.ndarray: ...


class Evaluation(Protocol):
    """Higher-order candidate-evaluation plan.

    Built by a factory (e.g. :func:`holdout`) that closes over the data /
    metric / split / weighting. Given a strategy, runs the search and
    yields :class:`Step` s.
    """

    def __call__(
        self,
        strategy: Strategy,
        *,
        max_dim: int,
        threshold: float = 0.0,
        filter: FilterFn | None = None,
    ) -> Iterator[Step]: ...


type AggregateFunc = Callable[[np.ndarray], float]
"""``(per-fold scores) -> scalar``."""


def ensure_2d_features(X: np.ndarray, *, name: str) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {X.ndim}D with shape {X.shape}")
    return X


def ensure_2d_target(Y: np.ndarray, *, name: str) -> np.ndarray:
    if Y.ndim == 1:
        return Y[:, None]
    if Y.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got {Y.ndim}D with shape {Y.shape}")
    return Y


def predict_2d(
    indices: Sequence[int],
    *,
    X_train: np.ndarray,
    X_query: np.ndarray,
    Y_train: np.ndarray,
    predict: PredictFunc,
) -> np.ndarray:
    idx = list(indices)
    predictions = predict(X_train[:, idx], Y_train, X_query[:, idx])
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    return predictions


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
    """Softmax weights that up-weight *low*-scoring folds.

    Intended for :func:`weighted_folds` under a higher-is-better metric:
    folds where the parent path scored poorly receive more weight in
    evaluating candidate subsets. With ``state=zeros`` the returned
    weights are uniform.

    Parameters
    ----------
    temperature : float
        Concentration parameter. Large values flatten toward uniform;
        small values concentrate weight on the worst fold.

    Raises
    ------
    ValueError
        If ``temperature`` is not positive.
    """
    validate_temperature(temperature)

    def fn(state: np.ndarray) -> np.ndarray:
        return softmax(state, temperature=temperature, maximize=True)

    return fn


def softmax_loss_weight(*, temperature: float = 1.0) -> WeightFunc:
    """Softmax weights that up-weight *high*-loss samples.

    Intended for :func:`weighted_timepoints`. With ``state=zeros`` the
    returned weights are uniform.

    Parameters
    ----------
    temperature : float
        Concentration parameter. Large values flatten toward uniform;
        small values concentrate weight on the worst sample.

    Raises
    ------
    ValueError
        If ``temperature`` is not positive.
    """
    validate_temperature(temperature)

    def fn(state: np.ndarray) -> np.ndarray:
        return softmax(state, temperature=temperature, maximize=False)

    return fn


# ---------------------------------------------------------------------------
# Per-sample loss helpers
# ---------------------------------------------------------------------------


def as_matching_2d(
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
    """Mean absolute error per sample, averaged across target dimensions."""
    predictions, observations = as_matching_2d(predictions, observations)
    return np.abs(predictions - observations).mean(axis=1)


def mean_squared_error_per_sample(
    predictions: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    """Mean squared error per sample, averaged across target dimensions."""
    predictions, observations = as_matching_2d(predictions, observations)
    return ((predictions - observations) ** 2).mean(axis=1)


def mean_negative_correlation_contribution_per_sample(
    predictions: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    """Per-sample negative-correlation contribution (lower is better).

    Each sample receives the negative of its standardised covariance
    contribution, averaged across dimensions. Summing over samples
    recovers the Pearson-correlation numerator, so weighting these is
    comparable in intent to weighting fold-level correlation scores.
    """
    predictions, observations = as_matching_2d(predictions, observations)

    p_centered = predictions - predictions.mean(axis=0, keepdims=True)
    o_centered = observations - observations.mean(axis=0, keepdims=True)
    denom = np.sqrt((p_centered**2).sum(axis=0) * (o_centered**2).sum(axis=0))
    safe_denom = np.where(denom > 0, denom, 1.0)
    contributions = (p_centered * o_centered) / safe_denom
    contributions = np.where(denom > 0, contributions, 0.0)
    return -contributions.mean(axis=1)


# ---------------------------------------------------------------------------
# Evaluation factories
# ---------------------------------------------------------------------------


def holdout(
    *,
    X_train: np.ndarray,
    X_val: np.ndarray,
    Y_train: np.ndarray,
    Y_val: np.ndarray,
    predict: PredictFunc,
    metric: MetricFunc,
) -> Evaluation:
    """Single train/validation holdout.

    Parameters
    ----------
    X_train, X_val : np.ndarray of shape (N, M)
    Y_train, Y_val : np.ndarray of shape (N,) or (N, D)
    predict : PredictFunc
    metric : MetricFunc
        Higher must be better.

    Returns
    -------
    Evaluation
        Higher-order: ``evaluation(strategy, *, max_dim, ...)``.
    """
    X_train_2d = ensure_2d_features(X_train, name="X_train")
    X_val_2d = ensure_2d_features(X_val, name="X_val")
    Y_train_2d = ensure_2d_target(Y_train, name="Y_train")
    Y_val_2d = ensure_2d_target(Y_val, name="Y_val")
    n_candidates = X_train_2d.shape[1]

    def score(indices: Sequence[int], parent_state: None) -> tuple[float, None]:
        del parent_state
        predictions = predict_2d(
            indices,
            X_train=X_train_2d, X_query=X_val_2d, Y_train=Y_train_2d,
            predict=predict,
        )
        return float(metric(predictions, Y_val_2d)), None

    def evaluation(
        strategy: Strategy,
        *,
        max_dim: int,
        threshold: float = 0.0,
        filter: FilterFn | None = None,
    ) -> Iterator[Step]:
        return strategy(
            score,
            n_candidates=n_candidates,
            initial_state=None,
            max_dim=max_dim,
            threshold=threshold,
            filter=filter,
        )

    return evaluation


def folds(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    folds: list[Fold],
    predict: PredictFunc,
    metric: MetricFunc,
    aggregate: AggregateFunc = lambda xs: float(np.mean(xs)),
) -> Evaluation:
    """K-fold evaluation, aggregated per candidate subset.

    Parameters
    ----------
    X : np.ndarray of shape (T, M)
    Y : np.ndarray of shape (T,) or (T, D)
    folds : list[Fold]
    predict : PredictFunc
    metric : MetricFunc
    aggregate : AggregateFunc
        ``(per-fold scores) -> scalar``. Default is the arithmetic mean.

    Returns
    -------
    Evaluation
    """
    X_2d = ensure_2d_features(X, name="X")
    Y_2d = ensure_2d_target(Y, name="Y")
    if len(folds) == 0:
        raise ValueError("folds must contain at least one Fold")
    fold_list = list(folds)
    n_candidates = X_2d.shape[1]

    def score(indices: Sequence[int], parent_state: None) -> tuple[float, None]:
        del parent_state
        per_fold = np.empty(len(fold_list))
        for k, fold in enumerate(fold_list):
            predictions = predict_2d(
                indices,
                X_train=X_2d[fold.train],
                X_query=X_2d[fold.validation],
                Y_train=Y_2d[fold.train],
                predict=predict,
            )
            per_fold[k] = float(metric(predictions, Y_2d[fold.validation]))
        return float(aggregate(per_fold)), None

    def evaluation(
        strategy: Strategy,
        *,
        max_dim: int,
        threshold: float = 0.0,
        filter: FilterFn | None = None,
    ) -> Iterator[Step]:
        return strategy(
            score,
            n_candidates=n_candidates,
            initial_state=None,
            max_dim=max_dim,
            threshold=threshold,
            filter=filter,
        )

    return evaluation


def loo(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    metric: MetricFunc,
    tau: int,
) -> Evaluation:
    """Leave-one-out simplex projection.

    The Theiler exclusion window grows with embedding dimension as
    ``(len(indices) - 1) * tau`` to match the original ``greedy_loo``
    semantics.

    Parameters
    ----------
    X : np.ndarray of shape (T, M)
    Y : np.ndarray of shape (T,) or (T, D)
    metric : MetricFunc
    tau : int
        Time delay used in the embedding.

    Returns
    -------
    Evaluation
    """
    X_2d = ensure_2d_features(X, name="X")
    Y_2d = ensure_2d_target(Y, name="Y")
    if tau < 0:
        raise ValueError(f"tau must be non-negative, got {tau}")
    n_candidates = X_2d.shape[1]

    def score(indices: Sequence[int], parent_state: None) -> tuple[float, None]:
        del parent_state
        idx = list(indices)
        theiler_window = (len(idx) - 1) * tau
        predictions = simplex_loo(X_2d[:, idx], Y_2d, theiler_window=theiler_window)
        if predictions.ndim == 1:
            predictions = predictions[:, None]
        return float(metric(predictions, Y_2d)), None

    def evaluation(
        strategy: Strategy,
        *,
        max_dim: int,
        threshold: float = 0.0,
        filter: FilterFn | None = None,
    ) -> Iterator[Step]:
        return strategy(
            score,
            n_candidates=n_candidates,
            initial_state=None,
            max_dim=max_dim,
            threshold=threshold,
            filter=filter,
        )

    return evaluation


def weighted_folds(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    folds: list[Fold],
    predict: PredictFunc,
    metric: MetricFunc,
    weight_update: WeightFunc,
) -> Evaluation:
    """Adaptive: score by weighted improvement in per-fold metrics.

    Path state is the parent's per-fold scores ``(K,)``. The next state
    is the candidate's per-fold scores. Score is
    ``weights @ (new_fold_scores - parent_fold_scores)`` where
    ``weights = weight_update(parent_state)``. With initial state of
    zeros and :func:`softmax_weight` (which yields uniform on zeros), the
    first step reduces to the mean per-fold score.

    Parameters
    ----------
    X, Y, folds, predict, metric : see :func:`folds`
    weight_update : WeightFunc
        Maps parent per-fold scores to weights of shape ``(K,)``.

    Returns
    -------
    Evaluation
    """
    X_2d = ensure_2d_features(X, name="X")
    Y_2d = ensure_2d_target(Y, name="Y")
    if len(folds) == 0:
        raise ValueError("folds must contain at least one Fold")
    fold_list = list(folds)
    n_candidates = X_2d.shape[1]
    initial_state = np.zeros(len(fold_list))

    def fold_scores(indices: Sequence[int]) -> np.ndarray:
        per = np.empty(len(fold_list))
        for k, fold in enumerate(fold_list):
            predictions = predict_2d(
                indices,
                X_train=X_2d[fold.train],
                X_query=X_2d[fold.validation],
                Y_train=Y_2d[fold.train],
                predict=predict,
            )
            per[k] = float(metric(predictions, Y_2d[fold.validation]))
        return per

    def score(
        indices: Sequence[int], parent_state: np.ndarray
    ) -> tuple[float, np.ndarray]:
        new = fold_scores(indices)
        weights = weight_update(parent_state)
        improvement = float(weights @ (new - parent_state))
        return improvement, new

    def evaluation(
        strategy: Strategy,
        *,
        max_dim: int,
        threshold: float = 0.0,
        filter: FilterFn | None = None,
    ) -> Iterator[Step]:
        return strategy(
            score,
            n_candidates=n_candidates,
            initial_state=initial_state,
            max_dim=max_dim,
            threshold=threshold,
            filter=filter,
        )

    return evaluation


def weighted_timepoints(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    folds: list[Fold],
    predict: PredictFunc,
    metric: MetricFunc,
    loss: SampleLossFn,
    weight_update: WeightFunc,
) -> Evaluation:
    """Adaptive: score by weighted reduction in per-sample losses.

    Path state is the parent's concatenated per-sample losses
    ``(N_total,)`` where ``N_total = sum(len(f.validation) for f in folds)``.
    Score is ``weights @ (parent_state - new_losses)`` where
    ``weights = weight_update(parent_state)``. With initial state of
    zeros and :func:`softmax_loss_weight`, the first step is proportional
    to ``-mean(losses)``.

    Parameters
    ----------
    X, Y, folds, predict, metric : see :func:`folds`
    loss : SampleLossFn
        Per-sample lower-is-better loss aligned with ``metric``.
    weight_update : WeightFunc
        Maps parent per-sample losses to weights of shape ``(N_total,)``.

    Returns
    -------
    Evaluation
    """
    X_2d = ensure_2d_features(X, name="X")
    Y_2d = ensure_2d_target(Y, name="Y")
    if len(folds) == 0:
        raise ValueError("folds must contain at least one Fold")
    fold_list = list(folds)
    n_candidates = X_2d.shape[1]
    n_total = sum(len(f.validation) for f in fold_list)
    initial_state = np.zeros(n_total)

    def sample_losses(indices: Sequence[int]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for fold in fold_list:
            predictions = predict_2d(
                indices,
                X_train=X_2d[fold.train],
                X_query=X_2d[fold.validation],
                Y_train=Y_2d[fold.train],
                predict=predict,
            )
            chunks.append(loss(predictions, Y_2d[fold.validation]))
        return np.concatenate(chunks)

    def score(
        indices: Sequence[int], parent_state: np.ndarray
    ) -> tuple[float, np.ndarray]:
        new = sample_losses(indices)
        weights = weight_update(parent_state)
        improvement = float(weights @ (parent_state - new))
        return improvement, new

    def evaluation(
        strategy: Strategy,
        *,
        max_dim: int,
        threshold: float = 0.0,
        filter: FilterFn | None = None,
    ) -> Iterator[Step]:
        return strategy(
            score,
            n_candidates=n_candidates,
            initial_state=initial_state,
            max_dim=max_dim,
            threshold=threshold,
            filter=filter,
        )

    return evaluation
