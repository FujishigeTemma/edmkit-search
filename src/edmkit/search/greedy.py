"""Greedy forward search over batched frontier evaluations."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from . import FilterFn, Step
from .runtime import FrontierEvaluation


def _available_candidates(
    n_candidates: int,
    selected: tuple[int, ...],
    *,
    filter: FilterFn | None,
) -> np.ndarray:
    blocked = set(selected)
    return np.asarray(
        [
            candidate
            for candidate in range(n_candidates)
            if candidate not in blocked and (filter is None or filter(candidate))
        ],
        dtype=np.intp,
    )


def greedy(
    evaluation: FrontierEvaluation,
    *,
    max_dim: int,
    threshold: float = 0.0,
    filter: FilterFn | None = None,
) -> Iterator[Step]:
    if evaluation.n_candidates < 1:
        raise ValueError(
            f"n_candidates must be >= 1, got {evaluation.n_candidates}"
        )
    if max_dim > evaluation.n_candidates:
        raise ValueError(
            f"max_dim must be <= n_candidates (={evaluation.n_candidates}), got {max_dim}"
        )

    path = evaluation.initial_path()

    for _ in range(max_dim):
        candidates = _available_candidates(
            evaluation.n_candidates,
            path.selected,
            filter=filter,
        )
        if len(candidates) == 0:
            return

        frontier = evaluation.evaluate_frontier([path], [candidates])
        frontier = [scored for scored in frontier if scored.score >= threshold]
        if not frontier:
            return

        chosen = max(frontier, key=lambda scored: scored.score)
        path = evaluation.advance(path, chosen)
        yield Step(
            index=chosen.candidate,
            score=chosen.score,
            selected=path.selected,
        )
