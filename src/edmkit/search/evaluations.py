"""Candidate-evaluation factories built on a fixed-chunk batch runtime."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any, Protocol

import numpy as np
from edmkit.metrics import MetricFunc
from edmkit.simplex_projection import loo as simplex_loo
from edmkit.splits import Fold
from edmkit.types import PredictFunc
from opentelemetry import trace

from . import FilterFn, Step, Strategy
from .runtime import (
    DEFAULT_EXTENSION_CHUNK_SIZE,
    FrontierEvaluation,
    ScoredExtension,
    SearchPath,
    SplitKernelData,
    extension_columns,
    flatten_frontier,
    make_split_kernel_data,
    predict_batch,
)


class SampleLossFn(Protocol):
    __name__: str

    def __call__(
        self,
        predictions: np.ndarray,
        observations: np.ndarray,
        /,
    ) -> np.ndarray: ...


class WeightFunc(Protocol):
    def __call__(self, state: np.ndarray) -> np.ndarray: ...


class Evaluation(Protocol):
    def __call__(
        self,
        strategy: Strategy,
        *,
        max_dim: int,
        threshold: float = 0.0,
        filter: FilterFn | None = None,
    ) -> Iterator[Step]: ...


type AggregateFunc = Callable[[np.ndarray], float]


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
    validate_temperature(temperature)

    def fn(state: np.ndarray) -> np.ndarray:
        return softmax(state, temperature=temperature, maximize=True)

    return fn


def softmax_loss_weight(*, temperature: float = 1.0) -> WeightFunc:
    validate_temperature(temperature)

    def fn(state: np.ndarray) -> np.ndarray:
        return softmax(state, temperature=temperature, maximize=False)

    return fn


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
    predictions, observations = as_matching_2d(predictions, observations)
    return np.abs(predictions - observations).mean(axis=1)


def mean_squared_error_per_sample(
    predictions: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    predictions, observations = as_matching_2d(predictions, observations)
    return ((predictions - observations) ** 2).mean(axis=1)


def mean_negative_correlation_contribution_per_sample(
    predictions: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    predictions, observations = as_matching_2d(predictions, observations)

    p_centered = predictions - predictions.mean(axis=0, keepdims=True)
    o_centered = observations - observations.mean(axis=0, keepdims=True)
    denom = np.sqrt((p_centered**2).sum(axis=0) * (o_centered**2).sum(axis=0))
    safe_denom = np.where(denom > 0, denom, 1.0)
    contributions = (p_centered * o_centered) / safe_denom
    contributions = np.where(denom > 0, contributions, 0.0)
    return -contributions.mean(axis=1)


def evaluation_from_frontier(frontier: FrontierEvaluation) -> Evaluation:
    def evaluation(
        strategy: Strategy,
        *,
        max_dim: int,
        threshold: float = 0.0,
        filter: FilterFn | None = None,
    ) -> Iterator[Step]:
        return strategy(
            frontier,
            max_dim=max_dim,
            threshold=threshold,
            filter=filter,
        )

    return evaluation


class CallbackEvaluation(FrontierEvaluation):
    def __init__(
        self,
        *,
        n_candidates: int,
        initial_value_state: Any,
        score_indices: Callable[[Sequence[int], Any], tuple[float, Any]],
    ):
        self.n_candidates = n_candidates
        self.initial_value_state = initial_value_state
        self.score_indices = score_indices

    def initial_path(self) -> SearchPath:
        return SearchPath(
            selected=(),
            value_state=self.initial_value_state,
            score=float("-inf"),
        )

    def evaluate_frontier(
        self,
        paths: Sequence[SearchPath],
        candidate_lists: Sequence[np.ndarray],
    ) -> list[ScoredExtension]:
        path_ids, candidates = flatten_frontier(candidate_lists)
        if len(candidates) == 0:
            return []

        dim = len(paths[int(path_ids[0])].selected) + 1
        with trace.get_tracer(__name__).start_as_current_span(
            "search.frontier",
            attributes={
                "dim": dim,
                "n_extensions": len(candidates),
                "chunk_size": 0,
                "n_chunks": 1,
                "n_paths": len(paths),
                "n_splits": 0,
                "evaluation_kind": "callback",
            },
        ) as span:
            results: list[ScoredExtension] = []
            for path_index, candidate in zip(path_ids, candidates, strict=True):
                path = paths[int(path_index)]
                score, next_value_state = self.score_indices(
                    path.selected + (int(candidate),),
                    path.value_state,
                )
                results.append(
                    ScoredExtension(
                        path_index=int(path_index),
                        candidate=int(candidate),
                        score=float(score),
                        next_value_state=next_value_state,
                    )
                )
            span.set_attribute("n_results", len(results))
            return results

    def advance(
        self,
        path: SearchPath,
        scored: ScoredExtension,
    ) -> SearchPath:
        return SearchPath(
            selected=path.selected + (scored.candidate,),
            value_state=scored.next_value_state,
            score=scored.score,
        )


class PredictFrontierEvaluation(FrontierEvaluation):
    def __init__(
        self,
        *,
        splits: Sequence[SplitKernelData],
        predict: PredictFunc,
        n_candidates: int,
        chunk_size: int = DEFAULT_EXTENSION_CHUNK_SIZE,
    ):
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self.splits = tuple(splits)
        self.predict = predict
        self.n_candidates = n_candidates
        self.chunk_size = chunk_size

    def initial_value_state(self) -> Any:
        return None

    def initial_path(self) -> SearchPath:
        return SearchPath(
            selected=(),
            value_state=self.initial_value_state(),
            score=float("-inf"),
        )

    def evaluate_frontier(
        self,
        paths: Sequence[SearchPath],
        candidate_lists: Sequence[np.ndarray],
    ) -> list[ScoredExtension]:
        path_ids, candidates = flatten_frontier(candidate_lists)
        if len(candidates) == 0:
            return []

        dim = len(paths[int(path_ids[0])].selected) + 1
        n_chunks = (len(candidates) + self.chunk_size - 1) // self.chunk_size
        with trace.get_tracer(__name__).start_as_current_span(
            "search.frontier",
            attributes={
                "dim": dim,
                "n_extensions": len(candidates),
                "chunk_size": self.chunk_size,
                "n_chunks": n_chunks,
                "n_paths": len(paths),
                "n_splits": len(self.splits),
            },
        ) as span:
            results: list[ScoredExtension] = []
            for start in range(0, len(candidates), self.chunk_size):
                stop = min(len(candidates), start + self.chunk_size)
                batch_path_ids = path_ids[start:stop]
                batch_candidates = candidates[start:stop]
                columns = extension_columns(paths, batch_path_ids, batch_candidates)
                split_predictions = [
                    predict_batch(split, columns, predict=self.predict)
                    for split in self.splits
                ]
                results.extend(
                    self.reduce_batch(
                        paths,
                        batch_path_ids,
                        batch_candidates,
                        split_predictions,
                    )
                )
            span.set_attribute("n_results", len(results))
            return results

    def reduce_batch(
        self,
        paths: Sequence[SearchPath],
        path_ids: np.ndarray,
        candidates: np.ndarray,
        split_predictions: Sequence[np.ndarray],
    ) -> list[ScoredExtension]:
        raise NotImplementedError

    def advance(
        self,
        path: SearchPath,
        scored: ScoredExtension,
    ) -> SearchPath:
        return SearchPath(
            selected=path.selected + (scored.candidate,),
            value_state=scored.next_value_state,
            score=scored.score,
        )


class HoldoutEvaluation(PredictFrontierEvaluation):
    def __init__(
        self,
        *,
        split: SplitKernelData,
        predict: PredictFunc,
        n_candidates: int,
        metric: MetricFunc,
    ):
        super().__init__(
            splits=[split],
            predict=predict,
            n_candidates=n_candidates,
        )
        self.metric = metric

    def reduce_batch(
        self,
        paths: Sequence[SearchPath],
        path_ids: np.ndarray,
        candidates: np.ndarray,
        split_predictions: Sequence[np.ndarray],
    ) -> list[ScoredExtension]:
        del paths
        split = self.splits[0]
        predictions = split_predictions[0]
        return [
            ScoredExtension(
                path_index=int(path_id),
                candidate=int(candidate),
                score=float(self.metric(predictions[i], split.y_query)),
                next_value_state=None,
            )
            for i, (path_id, candidate) in enumerate(
                zip(path_ids, candidates, strict=True)
            )
        ]


class FoldsEvaluation(PredictFrontierEvaluation):
    def __init__(
        self,
        *,
        splits: Sequence[SplitKernelData],
        predict: PredictFunc,
        n_candidates: int,
        metric: MetricFunc,
        aggregate: AggregateFunc,
    ):
        super().__init__(
            splits=splits,
            predict=predict,
            n_candidates=n_candidates,
        )
        self.metric = metric
        self.aggregate = aggregate

    def fold_scores_for(
        self,
        split_predictions: Sequence[np.ndarray],
        batch_index: int,
    ) -> np.ndarray:
        per_fold = np.empty(len(self.splits))
        for split_index, split in enumerate(self.splits):
            per_fold[split_index] = float(
                self.metric(split_predictions[split_index][batch_index], split.y_query)
            )
        return per_fold

    def reduce_batch(
        self,
        paths: Sequence[SearchPath],
        path_ids: np.ndarray,
        candidates: np.ndarray,
        split_predictions: Sequence[np.ndarray],
    ) -> list[ScoredExtension]:
        del paths
        results: list[ScoredExtension] = []
        for i, (path_id, candidate) in enumerate(
            zip(path_ids, candidates, strict=True)
        ):
            per_fold = self.fold_scores_for(split_predictions, i)
            results.append(
                ScoredExtension(
                    path_index=int(path_id),
                    candidate=int(candidate),
                    score=float(self.aggregate(per_fold)),
                    next_value_state=None,
                )
            )
        return results


class WeightedFoldsEvaluation(FoldsEvaluation):
    def __init__(
        self,
        *,
        splits: Sequence[SplitKernelData],
        predict: PredictFunc,
        n_candidates: int,
        metric: MetricFunc,
        weight_update: WeightFunc,
    ):
        super().__init__(
            splits=splits,
            predict=predict,
            n_candidates=n_candidates,
            metric=metric,
            aggregate=lambda xs: float(np.mean(xs)),
        )
        self.weight_update = weight_update

    def initial_value_state(self) -> Any:
        return np.zeros(len(self.splits), dtype=np.float64)

    def reduce_batch(
        self,
        paths: Sequence[SearchPath],
        path_ids: np.ndarray,
        candidates: np.ndarray,
        split_predictions: Sequence[np.ndarray],
    ) -> list[ScoredExtension]:
        results: list[ScoredExtension] = []
        for i, (path_id, candidate) in enumerate(
            zip(path_ids, candidates, strict=True)
        ):
            per_fold = self.fold_scores_for(split_predictions, i)
            parent_state = np.asarray(paths[int(path_id)].value_state, dtype=np.float64)
            weights = self.weight_update(parent_state)
            results.append(
                ScoredExtension(
                    path_index=int(path_id),
                    candidate=int(candidate),
                    score=float(weights @ (per_fold - parent_state)),
                    next_value_state=per_fold,
                )
            )
        return results


class WeightedTimepointsEvaluation(PredictFrontierEvaluation):
    def __init__(
        self,
        *,
        splits: Sequence[SplitKernelData],
        predict: PredictFunc,
        n_candidates: int,
        loss: SampleLossFn,
        weight_update: WeightFunc,
    ):
        super().__init__(
            splits=splits,
            predict=predict,
            n_candidates=n_candidates,
        )
        self.loss = loss
        self.weight_update = weight_update
        self.n_total = sum(split.query_size for split in self.splits)

    def initial_value_state(self) -> Any:
        return np.zeros(self.n_total, dtype=np.float64)

    def sample_losses_for(
        self,
        split_predictions: Sequence[np.ndarray],
        batch_index: int,
    ) -> np.ndarray:
        return np.concatenate(
            [
                self.loss(split_predictions[split_index][batch_index], split.y_query)
                for split_index, split in enumerate(self.splits)
            ]
        )

    def reduce_batch(
        self,
        paths: Sequence[SearchPath],
        path_ids: np.ndarray,
        candidates: np.ndarray,
        split_predictions: Sequence[np.ndarray],
    ) -> list[ScoredExtension]:
        results: list[ScoredExtension] = []
        for i, (path_id, candidate) in enumerate(
            zip(path_ids, candidates, strict=True)
        ):
            new_losses = self.sample_losses_for(split_predictions, i)
            parent_state = np.asarray(paths[int(path_id)].value_state, dtype=np.float64)
            weights = self.weight_update(parent_state)
            results.append(
                ScoredExtension(
                    path_index=int(path_id),
                    candidate=int(candidate),
                    score=float(weights @ (parent_state - new_losses)),
                    next_value_state=new_losses,
                )
            )
        return results


def holdout(
    *,
    X_train: np.ndarray,
    X_val: np.ndarray,
    Y_train: np.ndarray,
    Y_val: np.ndarray,
    predict: PredictFunc,
    metric: MetricFunc,
) -> Evaluation:
    X_train_2d = ensure_2d_features(X_train, name="X_train")
    X_val_2d = ensure_2d_features(X_val, name="X_val")
    Y_train_2d = ensure_2d_target(Y_train, name="Y_train")
    Y_val_2d = ensure_2d_target(Y_val, name="Y_val")

    return evaluation_from_frontier(
        HoldoutEvaluation(
            split=make_split_kernel_data(X_train_2d, X_val_2d, Y_train_2d, Y_val_2d),
            predict=predict,
            n_candidates=X_train_2d.shape[1],
            metric=metric,
        )
    )


def folds(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    folds: list[Fold],
    predict: PredictFunc,
    metric: MetricFunc,
    aggregate: AggregateFunc = lambda xs: float(np.mean(xs)),
) -> Evaluation:
    X_2d = ensure_2d_features(X, name="X")
    Y_2d = ensure_2d_target(Y, name="Y")
    if len(folds) == 0:
        raise ValueError("folds must contain at least one Fold")
    fold_list = list(folds)

    return evaluation_from_frontier(
        FoldsEvaluation(
            splits=[
                make_split_kernel_data(
                    X_2d[fold.train],
                    X_2d[fold.validation],
                    Y_2d[fold.train],
                    Y_2d[fold.validation],
                )
                for fold in fold_list
            ],
            predict=predict,
            n_candidates=X_2d.shape[1],
            metric=metric,
            aggregate=aggregate,
        )
    )


def loo(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    metric: MetricFunc,
    tau: int,
) -> Evaluation:
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

    return evaluation_from_frontier(
        CallbackEvaluation(
            n_candidates=n_candidates,
            initial_value_state=None,
            score_indices=score,
        )
    )


def weighted_folds(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    folds: list[Fold],
    predict: PredictFunc,
    metric: MetricFunc,
    weight_update: WeightFunc,
) -> Evaluation:
    X_2d = ensure_2d_features(X, name="X")
    Y_2d = ensure_2d_target(Y, name="Y")
    if len(folds) == 0:
        raise ValueError("folds must contain at least one Fold")
    fold_list = list(folds)

    return evaluation_from_frontier(
        WeightedFoldsEvaluation(
            splits=[
                make_split_kernel_data(
                    X_2d[fold.train],
                    X_2d[fold.validation],
                    Y_2d[fold.train],
                    Y_2d[fold.validation],
                )
                for fold in fold_list
            ],
            predict=predict,
            n_candidates=X_2d.shape[1],
            metric=metric,
            weight_update=weight_update,
        )
    )


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
    X_2d = ensure_2d_features(X, name="X")
    Y_2d = ensure_2d_target(Y, name="Y")
    if len(folds) == 0:
        raise ValueError("folds must contain at least one Fold")
    fold_list = list(folds)

    return evaluation_from_frontier(
        WeightedTimepointsEvaluation(
            splits=[
                make_split_kernel_data(
                    X_2d[fold.train],
                    X_2d[fold.validation],
                    Y_2d[fold.train],
                    Y_2d[fold.validation],
                )
                for fold in fold_list
            ],
            predict=predict,
            n_candidates=X_2d.shape[1],
            loss=loss,
            weight_update=weight_update,
        )
    )
