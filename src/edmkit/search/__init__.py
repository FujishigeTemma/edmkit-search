# ruff: noqa: F401
from collections.abc import Iterable

from .annealing import ScheduleFn, anneal, geometric_cooling
from .beam import beam
from .complementary import (
    WeightFunc,
    greedy_complementary_folds,
    greedy_complementary_timepoints,
    mean_abs_error_per_sample,
    softmax_loss_weight,
    softmax_weight,
)
from .convergence import causation, make_ccm_filter
from .dataset import Dataset, Subset, Transform
from .greedy import greedy
from .types import FilterFn, SampleLossFn, Selection, Step


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
