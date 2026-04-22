"""Public type contracts for ``edmkit.search``.

Three orthogonal concepts are made explicit:

* **Strategy** — how candidates are explored. Pure functions taking a
  scorer plus ``n_candidates`` / ``initial_state``. Implementations:
  :func:`~edmkit.search.greedy`, :func:`~edmkit.search.beam`.
* **Metric** — how prediction error is reduced to a scalar. Reused from
  :mod:`edmkit.metrics`.
* **Evaluation** — how one candidate subset is scored. A higher-order
  function that captures data / metric / split in a closure and awaits
  a strategy. Built by :mod:`~edmkit.search.evaluations` factories.

State is carried by the strategy through ``score``: every call returns
the state that would apply if the scored subset is selected. Static
evaluations thread ``None``; adaptive evaluations thread a per-path
``np.ndarray``.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import NamedTuple, Protocol

import numpy as np


type ScoreFunc[S] = Callable[[Sequence[int], S], tuple[float, S]]
"""``(indices, parent_state) -> (score, next_state)``.

Pure. Given the candidate subset a strategy is considering and the
parent path's state, returns the subset's score and the state that
would apply if the subset is selected.
"""


class FilterFn(Protocol):
    """``(index) -> accept`` over individual candidate indices."""

    def __call__(self, index: int, /) -> bool: ...


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
    For ``weighted_timepoints`` the input is parent per-sample losses
    ``(N_total,)``. The returned weights share the input's shape and are
    expected to sum to one.
    """

    def __call__(self, state: np.ndarray) -> np.ndarray: ...


class Strategy(Protocol):
    """Exploration algorithm: ``greedy`` / ``beam`` / ....

    Strategies are polymorphic in the state type ``S``: they read the
    starting ``initial_state`` and call ``score`` to propose extensions;
    they do not interpret ``S`` themselves.
    """

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


class Evaluation(Protocol):
    """Higher-order candidate-evaluation plan.

    Built by a factory (e.g. :func:`~edmkit.search.holdout`) that closes
    over the data / metric / split / weighting. Given a strategy, runs
    the search and yields :class:`Step` s.
    """

    def __call__(
        self,
        strategy: Strategy,
        *,
        max_dim: int,
        threshold: float = 0.0,
        filter: FilterFn | None = None,
    ) -> Iterator[Step]: ...


class Step(NamedTuple):
    """Result of a single selection step along the chosen path.

    Parameters
    ----------
    index : int
        Variable selected at this step.
    score : float
        Score of the current selection (strategy-dependent).
    selected : tuple[int, ...]
        All selected variable indices, in selection order.
    """

    index: int
    score: float
    selected: tuple[int, ...]


class Selection(NamedTuple):
    """Aggregated result of a search run.

    Parameters
    ----------
    indices : list[int]
        Indices selected, in order.
    scores : list[float]
        Score produced at each dimension.
    """

    indices: list[int]
    scores: list[float]
