# ruff: noqa: E402, F401
"""Subset-selection search.

Three orthogonal concepts — combine freely:

* **Strategy** — :func:`greedy`, :func:`beam`. Pure search algorithms.
* **Metric** — reused from :mod:`edmkit.metrics`.
* **Evaluation** — higher-order functions built by
  :func:`holdout`, :func:`folds`, :func:`loo`, :func:`weighted_folds`,
  :func:`weighted_timepoints`. Capture data / metric / split /
  weighting in a closure; accept a strategy and return an iterator of
  :class:`Step` s.

Example
-------
>>> evaluation = holdout(
...     X_train=X_tr, X_val=X_va, Y_train=Y_tr, Y_val=Y_va,
...     predict=simplex_projection, metric=mean_rho,
... )
>>> steps = list(evaluation(greedy, max_dim=10, filter=my_filter))
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import NamedTuple, Protocol

# Core selection protocol — defined here before strategy modules import them
# so that ``from . import Step, ...`` inside greedy.py / beam.py can resolve
# via the partially-initialised package namespace.


type ScoreFunc[S] = Callable[[Sequence[int], S], tuple[float, S]]
"""``(indices, parent_state) -> (score, next_state)``."""


class FilterFn(Protocol):
    """``(index) -> accept`` over individual candidate indices."""

    def __call__(self, index: int, /) -> bool: ...


class Step(NamedTuple):
    """Result of a single selection step along the chosen path."""

    index: int
    score: float
    selected: tuple[int, ...]


class Selection(NamedTuple):
    """Aggregated result of a search run."""

    indices: list[int]
    scores: list[float]


class Strategy(Protocol):
    """Exploration algorithm: ``greedy`` / ``beam`` / ...."""

    def __call__[S](
        self,
        score: ScoreFunc[S],
        *,
        n_candidates: int,
        initial_state: S,
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
    """Collect :class:`Step` s into a :class:`Selection`.

    Indices come from the last step's ``selected`` tuple; scores are the
    per-step scores in order.
    """
    steps_list = list(steps)
    if not steps_list:
        return Selection(indices=[], scores=[])
    return Selection(
        indices=list(steps_list[-1].selected),
        scores=[s.score for s in steps_list],
    )
