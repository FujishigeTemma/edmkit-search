from collections.abc import Iterator
from functools import partial
from typing import Callable

import numpy as np

from mde.dataset import Dataset, Subset
from mde.types import FilterFn, MetricFn, PredictFn

from .common import prepare_data, score_subset
from .types import Step

ScheduleFn = Callable[[int, int], float]
"""Temperature schedule: (step, n_steps) -> temperature."""


def geometric_cooling(*, T_start: float = 1.0, T_end: float = 0.01) -> ScheduleFn:
    """Create a geometric cooling schedule.

    Temperature decays exponentially from ``T_start`` to ``T_end``.

    Parameters
    ----------
    T_start : float
        Initial temperature. Default is 1.0.
    T_end : float
        Final temperature. Default is 0.01.

    Returns
    -------
    ScheduleFn
        ``(step, n_steps) -> temperature``.

    Raises
    ------
    ValueError
        If T_start or T_end are not positive, or T_start <= T_end.
    """
    if T_start <= 0:
        raise ValueError(f"T_start must be positive, got {T_start}")
    if T_end <= 0:
        raise ValueError(f"T_end must be positive, got {T_end}")
    if T_start <= T_end:
        raise ValueError(
            f"T_start must be > T_end, got T_start={T_start}, T_end={T_end}"
        )

    ratio = T_end / T_start

    def schedule(step: int, n_steps: int) -> float:
        if n_steps <= 1:
            return T_end
        return T_start * ratio ** (step / (n_steps - 1))

    return schedule


def perturb(
    current: list[int],
    *,
    available_list: list[int],
    max_dim: int,
    rng: np.random.Generator,
    filter: FilterFn | None,
    X_train: np.ndarray,
    Y_train: np.ndarray,
) -> list[int] | None:
    """Propose a neighbor state by add, remove, or swap.

    Returns
    -------
    list[int] | None
        Proposed indices, or None if no valid perturbation found.
    """

    # Build list of possible moves
    moves: list[str] = []
    if len(current) < max_dim and available_list:
        moves.append("add")
    if len(current) > 1:
        moves.append("remove")
    if available_list and len(current) >= 1:
        moves.append("swap")

    if not moves:
        return None

    rng.shuffle(moves)

    for move in moves:
        if move == "add":
            order = rng.permutation(len(available_list))
            for i in order:
                idx = available_list[i]
                if filter is not None and not filter(X_train[:, idx], Y_train):
                    continue
                return current + [idx]

        elif move == "remove":
            pos = int(rng.integers(len(current)))
            return current[:pos] + current[pos + 1 :]

        elif move == "swap":
            pos = int(rng.integers(len(current)))
            order = rng.permutation(len(available_list))
            for i in order:
                idx = available_list[i]
                if filter is not None and not filter(X_train[:, idx], Y_train):
                    continue
                result = current.copy()
                result[pos] = idx
                return result

    return None


def anneal(
    train: Dataset | Subset,
    validation: Dataset | Subset,
    *,
    predict: PredictFn,
    metric: MetricFn,
    n_steps: int = 1000,
    max_dim: int = 10,
    schedule: ScheduleFn | None = None,
    filter: FilterFn | None = None,
    rng: np.random.Generator | None = None,
) -> Iterator[Step]:
    """Select variables via simulated annealing.

    Phase 1 explores the variable space stochastically, then Phase 2
    orders the found variables by greedy forward selection and yields
    one ``Step`` per variable.

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
    n_steps : int
        Number of SA iterations. Default is 1000.
    max_dim : int
        Maximum number of variables to select. Default is 10.
    schedule : ScheduleFn | None
        Temperature schedule. Defaults to ``geometric_cooling()``.
    filter : FilterFn | None
        Optional filter ``(x, Y) -> bool`` to accept/reject candidates.
    rng : np.random.Generator | None
        Random number generator for reproducibility.

    Yields
    ------
    Step
        Result of each ordered variable in the found set.
    """
    X_train, X_validation, Y_train, Y_validation = prepare_data(train, validation)

    N = X_train.shape[1]
    if max_dim > N:
        raise ValueError(f"max_dim must be <= N (={N}), got {max_dim}")

    if schedule is None:
        schedule = geometric_cooling()
    if rng is None:
        rng = np.random.default_rng()

    score = partial(
        score_subset,
        X_train=X_train,
        X_validation=X_validation,
        Y_train=Y_train,
        Y_validation=Y_validation,
        predict=predict,
        metric=metric,
    )

    # --- Phase 1: Find initial single best variable ---
    best_single_idx = -1
    best_single_score = float("-inf")
    for i in range(N):
        if filter is not None and not filter(X_train[:, i], Y_train):
            continue
        s = score([i])
        if s > best_single_score:
            best_single_score = s
            best_single_idx = i

    if best_single_idx == -1:
        return

    current = [best_single_idx]
    current_score = best_single_score
    available = set(range(N)) - {best_single_idx}
    available_list = sorted(available)
    best = list(current)
    best_score = current_score

    # --- Phase 1: SA exploration ---
    for step in range(n_steps):
        T = schedule(step, n_steps)

        proposal = perturb(
            current,
            available_list=available_list,
            max_dim=max_dim,
            rng=rng,
            filter=filter,
            X_train=X_train,
            Y_train=Y_train,
        )
        if proposal is None:
            continue

        proposal_score = score(proposal)
        delta = proposal_score - current_score

        if delta > 0 or (T > 0 and rng.random() < np.exp(delta / T)):
            current_set = set(current)
            proposal_set = set(proposal)
            added = proposal_set - current_set
            removed = current_set - proposal_set
            available = (available | removed) - added
            for idx in removed:
                available_list.append(idx)
            for idx in added:
                available_list.remove(idx)
            current = proposal
            current_score = proposal_score

            if current_score > best_score:
                best = list(current)
                best_score = current_score

    # --- Phase 2: Order the found set by greedy forward selection ---
    remaining = set(best)
    ordered: list[int] = []

    while remaining:
        best_idx = -1
        best_s = float("-inf")
        for idx in remaining:
            ordered.append(idx)
            s = score(ordered)
            ordered.pop()
            if s > best_s:
                best_s = s
                best_idx = idx

        if best_idx == -1:
            break

        ordered.append(best_idx)
        remaining.remove(best_idx)
        yield Step(
            index=best_idx,
            score=best_s,
            selected=tuple(ordered),
        )
