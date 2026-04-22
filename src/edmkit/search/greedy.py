"""Greedy forward search.

Pure function. Takes a scorer plus ``n_candidates`` / ``initial_state`` /
``max_dim`` and yields :class:`Step` s. Equivalent to :func:`beam` with
``beam_width=1``.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence

from . import FilterFn, ScoreFunc, Step


def greedy[S](
    score: ScoreFunc[S],
    *,
    n_candidates: int,
    initial_state: S,
    max_dim: int,
    threshold: float = 0.0,
    filter: FilterFn | None = None,
) -> Iterator[Step]:
    """Greedily select up to ``max_dim`` indices from ``range(n_candidates)``.

    At each step the highest-scoring candidate (per ``score``) that passes
    ``filter`` and clears ``threshold`` is appended. Iteration ends early
    when no candidate qualifies.

    Parameters
    ----------
    score : ScoreFunc[S]
        Pure ``(indices, parent_state) -> (score, next_state)``. Usually
        produced by an :class:`~edmkit.search.types.Evaluation` factory.
    n_candidates : int
        Number of available candidate columns; the candidate set is
        ``range(n_candidates)``.
    initial_state : S
        State of a new path before any selection. ``None`` for static
        evaluations; e.g. ``np.zeros(K)`` for adaptive ones.
    max_dim : int
        Maximum number of selections. Must be ``<= n_candidates``.
    threshold : float
        Candidates with score strictly below this are skipped.
    filter : FilterFn | None
        Per-index predicate over candidate columns.

    Yields
    ------
    Step

    Raises
    ------
    ValueError
        If ``n_candidates < 1`` or ``max_dim > n_candidates``.
    """
    if n_candidates < 1:
        raise ValueError(f"n_candidates must be >= 1, got {n_candidates}")
    if max_dim > n_candidates:
        raise ValueError(
            f"max_dim must be <= n_candidates (={n_candidates}), got {max_dim}"
        )

    state: S = initial_state
    selected: list[int] = []
    available: set[int] = set(range(n_candidates))

    for _ in range(max_dim):
        best: tuple[int, float, S] | None = None
        for candidate in available:
            if filter is not None and not filter(candidate):
                continue
            extension: Sequence[int] = selected + [candidate]
            cand_score, cand_state = score(extension, state)
            if cand_score < threshold:
                continue
            if best is None or cand_score > best[1]:
                best = (candidate, cand_score, cand_state)

        if best is None:
            return

        chosen, chosen_score, state = best
        selected.append(chosen)
        available.remove(chosen)

        yield Step(
            index=chosen,
            score=chosen_score,
            selected=tuple(selected),
        )
