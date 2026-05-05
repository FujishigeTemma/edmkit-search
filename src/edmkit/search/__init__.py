# ruff: noqa: E402, F401
"""Subset-selection search with a batch-first frontier runtime."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import NamedTuple, Protocol


class FilterFn(Protocol):
    def __call__(self, index: int, /) -> bool: ...


class Step(NamedTuple):
    index: int
    score: float
    selected: tuple[int, ...]


class Selection(NamedTuple):
    indices: list[int]
    scores: list[float]


class Strategy(Protocol):
    def __call__(
        self,
        evaluation,
        *,
        max_dim: int,
        threshold: float = 0.0,
        filter: FilterFn | None = None,
    ) -> Iterator[Step]: ...


from .beam import beam
from .dataset import Dataset, Subset, Transform
from .evaluations import (
    Evaluation,
    SampleLossFn,
    WeightFunc,
    folds,
    holdout,
    loo,
    mean_abs_error_per_sample,
    mean_negative_correlation_contribution_per_sample,
    mean_squared_error_per_sample,
    softmax_loss_weight,
    softmax_weight,
    weighted_folds,
    weighted_timepoints,
)
from .greedy import greedy


def collect(steps: Iterable[Step]) -> Selection:
    steps_list = list(steps)
    if not steps_list:
        return Selection(indices=[], scores=[])
    return Selection(
        indices=list(steps_list[-1].selected),
        scores=[s.score for s in steps_list],
    )
