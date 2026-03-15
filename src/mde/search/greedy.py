from collections.abc import Iterator


from mde.dataset import Dataset, Subset
from mde.types import FilterFn, MetricFn, PredictFn

from .common import prepare_data, score_subset
from .types import Step


def greedy(
    train: Dataset | Subset,
    validation: Dataset | Subset,
    *,
    predict: PredictFn,
    metric: MetricFn,
    threshold: float = 0.0,
    max_dim: int = 10,
    filter: FilterFn | None = None,
) -> Iterator[Step]:
    """Greedily select variables that maximize prediction skill.

    Yields one ``Step`` per dimension, allowing early termination and
    custom control logic.

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
    filter : FilterFn | None
        Optional filter ``(x, Y) -> bool`` to accept/reject candidates.
        Called on the best candidate each iteration; if rejected, tries next best.

    Yields
    ------
    Step
        Result of each dimension selection step.
    """
    X_train, X_validation, Y_train, Y_validation = prepare_data(train, validation)

    N = X_train.shape[1]
    if max_dim > N:
        raise ValueError(f"max_dim must be <= N (={N}), got {max_dim}")

    available: set[int] = set(range(N))
    selected_indices: list[int] = []

    for dim in range(1, max_dim + 1):
        results = [
            (v, score_subset(
                selected_indices + [v],
                X_train=X_train,
                X_validation=X_validation,
                Y_train=Y_train,
                Y_validation=Y_validation,
                predict=predict,
                metric=metric,
            ))
            for v in available
        ]

        candidates = sorted(
            ((idx, s) for idx, s in results if s >= threshold),
            key=lambda r: r[1],
            reverse=True,
        )

        best = None
        for idx, score in candidates:
            if filter is not None and not filter(X_train[:, idx], Y_train):
                continue
            best = (idx, score)
            break

        if best is None:
            return

        best_idx, best_score = best
        selected_indices.append(best_idx)
        available.remove(best_idx)

        yield Step(
            index=best_idx,
            score=best_score,
            selected=tuple(selected_indices),
        )
