# ruff: noqa: F401
from collections.abc import Iterable

from .annealing import ScheduleFn, anneal, geometric_cooling
from .beam import beam
from .convergence import causation
from .dataset import (
    Dataset,
    Fold,
    Subset,
    Transform,
    expanding_splits,
    sliding_splits,
    temporal_split,
)
from .greedy import greedy
from .metrics import mae, mean_rho, negate, rmse
from .types import FilterFn, MetricFn, PredictFn, Selection, Step


def collect(steps: Iterable[Step]) -> Selection:
    """Collect Steps into a Selection.

    Takes indices from the last Step's ``selected`` field,
    scores from each Step's ``score``.

    Parameters
    ----------
    steps : Iterable[Step]
        Steps from any search algorithm.

    Returns
    -------
    Selection
        Aggregated result.
    """
    steps_list = list(steps)
    if not steps_list:
        return Selection(indices=[], scores=[])
    return Selection(
        indices=list(steps_list[-1].selected),
        scores=[s.score for s in steps_list],
    )
