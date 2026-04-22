"""Beam search.

Pure function. Maintains ``beam_width`` partial paths and explores all
extensions at each step. Each path carries its own ``state`` so adaptive
evaluations evolve independently across beams. ``beam_width=1`` reduces
to plain greedy.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from . import FilterFn, ScoreFunc, Step


@dataclass(frozen=True)
class BeamPath[S]:
    selected: tuple[int, ...]
    state: S
    score: float


def beam[S](
    score: ScoreFunc[S],
    *,
    n_candidates: int,
    initial_state: S,
    max_dim: int,
    beam_width: int,
    threshold: float = 0.0,
    filter: FilterFn | None = None,
) -> Iterator[Step]:
    """Beam-search up to ``max_dim`` indices from ``range(n_candidates)``.

    Maintains the top-``beam_width`` paths by score. Each path threads
    its own ``state`` through ``score`` so adaptive evaluations evolve
    independently across beams. Yields the best path's latest step at
    every dimension.

    Parameters
    ----------
    score : ScoreFunc[S]
    n_candidates : int
    initial_state : S
    max_dim : int
    beam_width : int
        Number of partial paths to keep. ``1`` reduces to greedy.
    threshold : float
    filter : FilterFn | None

    Yields
    ------
    Step

    Raises
    ------
    ValueError
        If ``n_candidates < 1``, ``beam_width < 1``, or
        ``max_dim > n_candidates``.
    """
    if n_candidates < 1:
        raise ValueError(f"n_candidates must be >= 1, got {n_candidates}")
    if max_dim > n_candidates:
        raise ValueError(
            f"max_dim must be <= n_candidates (={n_candidates}), got {max_dim}"
        )
    if beam_width < 1:
        raise ValueError(f"beam_width must be >= 1, got {beam_width}")

    paths: list[BeamPath[S]] = [
        BeamPath(selected=(), state=initial_state, score=float("-inf"))
    ]

    all_indices = set(range(n_candidates))

    for _ in range(max_dim):
        frontier: list[BeamPath[S]] = []
        for path in paths:
            for candidate in all_indices - set(path.selected):
                if filter is not None and not filter(candidate):
                    continue
                extension: Sequence[int] = list(path.selected) + [candidate]
                cand_score, cand_state = score(extension, path.state)
                if cand_score < threshold:
                    continue
                frontier.append(
                    BeamPath(
                        selected=path.selected + (candidate,),
                        state=cand_state,
                        score=cand_score,
                    )
                )

        if not frontier:
            return

        paths = sorted(frontier, key=lambda p: p.score, reverse=True)[:beam_width]
        top = paths[0]
        yield Step(
            index=top.selected[-1],
            score=top.score,
            selected=top.selected,
        )
