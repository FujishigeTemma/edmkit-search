from collections.abc import Iterator

from .dataset import Dataset, Subset
from .types import FilterFn, MetricFn, PredictFn

from .common import prepare_data, score_subset
from .types import Step


def beam(
    train: Dataset | Subset,
    validation: Dataset | Subset,
    *,
    predict: PredictFn,
    metric: MetricFn,
    threshold: float = 0.0,
    max_dim: int = 10,
    beam_width: int = 3,
    filter: FilterFn | None = None,
) -> Iterator[Step]:
    """Select variables via beam search.

    Maintains ``beam_width`` candidate paths at each step, exploring
    broader than greedy. ``beam_width=1`` is equivalent to greedy search.

    Parameters
    ----------
    train : Dataset | Subset
        Training data. Only ``.X`` and ``.Y`` are accessed.
    validation : Dataset | Subset
        Validation data for evaluating candidates.
    predict : PredictFn
        Prediction function ``(X_train, Y_train, X_query) -> predictions``.
    metric : MetricFn
        Metric function for evaluating prediction quality.
    threshold : float
        Minimum score for candidate selection. Default is 0.0.
    max_dim : int
        Maximum number of variables to select. Default is 10.
    beam_width : int
        Number of candidate paths to maintain. Default is 3.
    filter : FilterFn | None
        Optional filter ``(x, Y) -> bool`` to accept/reject candidates.

    Yields
    ------
    Step
        Best beam's result at each dimension step.
    """
    X_train, X_validation, Y_train, Y_validation = prepare_data(train, validation)

    N = X_train.shape[1]
    if max_dim > N:
        raise ValueError(f"max_dim must be <= N (={N}), got {max_dim}")
    if beam_width < 1:
        raise ValueError(f"beam_width must be >= 1, got {beam_width}")

    # Each beam is (indices, score)
    beams: list[tuple[list[int], float]] = [([], float("-inf"))]

    for dim in range(1, max_dim + 1):
        # Collect unique candidate subsets before scoring
        unique: dict[frozenset[int], list[int]] = {}
        for indices, _ in beams:
            used = set(indices)
            for candidate in range(N):
                if candidate in used:
                    continue
                if filter is not None and not filter(X_train[:, candidate], Y_train):
                    continue

                new_indices = indices + [candidate]
                key = frozenset(new_indices)
                if key not in unique:
                    unique[key] = new_indices

        if not unique:
            return

        # Score each unique subset once
        scored = [
            (indices, score_subset(
                indices,
                X_train=X_train,
                X_validation=X_validation,
                Y_train=Y_train,
                Y_validation=Y_validation,
                predict=predict,
                metric=metric,
            ))
            for indices in unique.values()
        ]

        # Filter by threshold and keep top beam_width
        above = [(idx, s) for idx, s in scored if s >= threshold]
        if not above:
            return

        beams = sorted(above, key=lambda b: b[1], reverse=True)[:beam_width]

        best_indices, best_score = beams[0]
        yield Step(
            index=best_indices[-1],
            score=best_score,
            selected=tuple(best_indices),
        )
