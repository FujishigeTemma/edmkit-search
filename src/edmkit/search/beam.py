"""Beam search over batched frontier evaluations."""
from __future__ import annotations

from collections.abc import Iterator

from . import FilterFn, Step
from .greedy import _available_candidates
from .runtime import FrontierEvaluation, SearchPath


def beam(
    evaluation: FrontierEvaluation,
    *,
    max_dim: int,
    beam_width: int,
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
    if beam_width < 1:
        raise ValueError(f"beam_width must be >= 1, got {beam_width}")

    paths: list[SearchPath] = [evaluation.initial_path()]

    for _ in range(max_dim):
        candidate_lists = [
            _available_candidates(
                evaluation.n_candidates,
                path.selected,
                filter=filter,
            )
            for path in paths
        ]
        if not any(len(candidate_list) > 0 for candidate_list in candidate_lists):
            return

        frontier = evaluation.evaluate_frontier(paths, candidate_lists)
        frontier = [scored for scored in frontier if scored.score >= threshold]
        if not frontier:
            return

        next_paths = [
            evaluation.advance(paths[scored.path_index], scored)
            for scored in frontier
        ]
        next_paths.sort(key=lambda path: path.score, reverse=True)
        paths = next_paths[:beam_width]

        top = paths[0]
        yield Step(
            index=top.selected[-1],
            score=top.score,
            selected=top.selected,
        )
