from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from opentelemetry import trace

DEFAULT_BATCH_DTYPE = np.dtype(np.float32)
DEFAULT_EXTENSION_CHUNK_SIZE = 10000


@dataclass(frozen=True)
class SplitKernelData:
    x_train_t: np.ndarray
    x_query_t: np.ndarray
    y_train: np.ndarray
    y_query: np.ndarray

    @property
    def n_candidates(self) -> int:
        return int(self.x_train_t.shape[0])

    @property
    def train_size(self) -> int:
        return int(self.x_train_t.shape[1])

    @property
    def query_size(self) -> int:
        return int(self.x_query_t.shape[1])

    @property
    def target_size(self) -> int:
        return int(self.y_train.shape[1])


@dataclass
class SearchPath:
    selected: tuple[int, ...]
    value_state: Any
    score: float


@dataclass(frozen=True)
class ScoredExtension:
    path_index: int
    candidate: int
    score: float
    next_value_state: Any


class FrontierEvaluation(Protocol):
    n_candidates: int

    def initial_path(self) -> SearchPath: ...

    def evaluate_frontier(
        self,
        paths: Sequence[SearchPath],
        candidate_lists: Sequence[np.ndarray],
    ) -> list[ScoredExtension]: ...

    def advance(
        self,
        path: SearchPath,
        scored: ScoredExtension,
    ) -> SearchPath: ...


def contiguous_2d(array: np.ndarray, *, dtype: np.dtype | None = None) -> np.ndarray:
    out = array[:, None] if array.ndim == 1 else array
    if out.ndim != 2:
        raise ValueError(f"expected 1D or 2D array, got shape {array.shape}")
    return np.ascontiguousarray(out.astype(dtype or out.dtype, copy=False))


def make_split_kernel_data(
    X_train: np.ndarray,
    X_query: np.ndarray,
    Y_train: np.ndarray,
    Y_query: np.ndarray,
    *,
    dtype: np.dtype = DEFAULT_BATCH_DTYPE,
) -> SplitKernelData:
    x_train = contiguous_2d(X_train, dtype=dtype)
    x_query = contiguous_2d(X_query, dtype=dtype)
    y_train = contiguous_2d(Y_train, dtype=dtype)
    y_query = contiguous_2d(Y_query, dtype=dtype)
    if x_train.shape[1] != x_query.shape[1]:
        raise ValueError(
            f"X_train and X_query must have the same number of columns, got "
            f"{x_train.shape[1]} and {x_query.shape[1]}"
        )
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"X_train and Y_train length mismatch: {x_train.shape[0]} vs {y_train.shape[0]}"
        )
    if y_train.shape[1] != y_query.shape[1]:
        raise ValueError(
            f"Y_train and Y_query must have the same number of columns, got "
            f"{y_train.shape[1]} and {y_query.shape[1]}"
        )
    return SplitKernelData(
        x_train_t=np.ascontiguousarray(x_train.T),
        x_query_t=np.ascontiguousarray(x_query.T),
        y_train=y_train,
        y_query=y_query,
    )


def flatten_frontier(
    candidate_lists: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    path_ids: list[np.ndarray] = []
    candidates: list[np.ndarray] = []
    for path_index, candidate_list in enumerate(candidate_lists):
        if len(candidate_list) == 0:
            continue
        candidate_array = np.asarray(candidate_list, dtype=np.intp)
        path_ids.append(np.full(len(candidate_array), path_index, dtype=np.intp))
        candidates.append(candidate_array)
    if not candidates:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    return np.concatenate(path_ids), np.concatenate(candidates)


def extension_columns(
    paths: Sequence[SearchPath],
    path_ids: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    if len(candidates) == 0:
        return np.empty((0, 0), dtype=np.intp)

    first_selected = paths[int(path_ids[0])].selected
    next_dim = len(first_selected) + 1
    columns = np.empty((len(candidates), next_dim), dtype=np.intp)

    for row, (path_id, candidate) in enumerate(zip(path_ids, candidates, strict=True)):
        selected = paths[int(path_id)].selected
        if len(selected) + 1 != next_dim:
            raise ValueError(
                f"all paths in a frontier step must have the same dimension, got "
                f"{next_dim - 1} and {len(selected)}"
            )
        if selected:
            columns[row, :-1] = selected
        columns[row, -1] = int(candidate)

    return columns


def predict_batch(
    split: SplitKernelData,
    columns: np.ndarray,
    *,
    predict,
) -> np.ndarray:
    batch = len(columns)
    embedding_dim = int(columns.shape[1]) if columns.ndim == 2 and batch else 0
    with trace.get_tracer(__name__).start_as_current_span(
        "search.predict_batch",
        attributes={
            "batch_size": batch,
            "embedding_dim": embedding_dim,
            "train_size": split.train_size,
            "query_size": split.query_size,
            "target_size": split.target_size,
        },
    ):
        if batch == 0:
            return np.empty(
                (0, split.query_size, split.target_size), dtype=split.y_train.dtype
            )

        x_train = np.swapaxes(split.x_train_t[columns], 1, 2)
        x_query = np.swapaxes(split.x_query_t[columns], 1, 2)
        y_train = np.broadcast_to(
            split.y_train[None, :, :],
            (batch, split.train_size, split.target_size),
        )

        predictions = np.asarray(predict(x_train, y_train, x_query))
        if predictions.ndim == 1:
            if batch != 1 or split.target_size != 1:
                raise ValueError(
                    f"batched predict returned 1D output for batch={batch}, "
                    f"target_size={split.target_size}"
                )
            predictions = predictions[None, :, None]
        elif predictions.ndim == 2:
            if (
                predictions.shape == (batch, split.query_size)
                and split.target_size == 1
            ):
                predictions = predictions[:, :, None]
            elif batch == 1 and predictions.shape == (
                split.query_size,
                split.target_size,
            ):
                predictions = predictions[None, :, :]
            else:
                raise ValueError(
                    f"batched predict returned 2D output with unexpected shape "
                    f"{predictions.shape}; expected ({batch}, {split.query_size}) or "
                    f"({split.query_size}, {split.target_size})"
                )
        elif predictions.ndim != 3:
            raise ValueError(
                f"batched predict must return a 3D array, got {predictions.ndim}D "
                f"with shape {predictions.shape}"
            )

        expected_shape = (batch, split.query_size, split.target_size)
        if predictions.shape != expected_shape:
            raise ValueError(
                f"batched predict returned shape {predictions.shape}, expected "
                f"{expected_shape}"
            )

        return predictions
