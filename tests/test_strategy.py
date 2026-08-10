from __future__ import annotations

import math
from functools import cache
from typing import NamedTuple

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from edmkit.search import state, strategy
from edmkit.search.energy import Contexts, Energies, Energy
from edmkit.search.neighborhood import Neighborhood
from edmkit.search.state import States
from edmkit.search.strategy import Frontier


class BeamProblem(NamedTuple):
    seed: int
    n: int
    width: int
    depth: int
    beams: int
    cutoff: float


class BeamCase(NamedTuple):
    seed: int
    n: int
    width: int
    depth: int
    beams: int
    cutoff: float


class GreedyProblem(NamedTuple):
    seed: int
    n: int
    depth: int
    cutoff: float


class GreedyCase(NamedTuple):
    seed: int
    n: int
    depth: int
    cutoff: float


def readonly(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


@cache
def node_energy(seed: int, row: tuple[int, ...]) -> float:
    """Deterministic pseudo-random node value shared by the test energy and the checks below."""
    return float(np.random.default_rng([seed, *row]).uniform(-1.0, 1.0))


def path_energy(seed: int, row: tuple[int, ...]) -> float:
    """The energy `accumulating_energy` assigns to a state: node values accumulated along its prefixes."""
    energy = 0.0
    for d in range(1, len(row) + 1):
        energy = node_energy(seed, row[:d]) + energy
    return energy


def accumulating_energy(seed: int) -> Energy:
    """Energy = the state's node value plus its parent's energy, carried through the context.

    Feeding the context back into the energy makes context routing observable from the outside: if a
    strategy hands E the wrong parent's context, the yielded energy no longer equals `path_energy` of
    the yielded state. Outputs are readonly: frontiers are immutable batches."""

    def E(states: States, contexts: Contexts) -> tuple[Energies, Contexts]:
        energies = np.array([node_energy(seed, tuple(row)) for row in states.tolist()]) + contexts[:, 0]
        return readonly(energies), readonly(energies[:, None].copy())

    return E


def ordered_forward(n: int) -> Neighborhood:
    """`neighborhood.forward` minus the per-parent shuffle: children appear in increasing index order, deterministically."""

    def expand(states: States, _rng: np.random.Generator) -> tuple[States, np.ndarray]:
        rows, parents = [], []
        for i, row in enumerate(states.tolist()):
            for j in range(n):
                if j not in row:
                    rows.append([*row, j])
                    parents.append(i)
        children = np.asarray(rows, dtype=np.int64).reshape(len(rows), states.shape[1] + 1)
        return children, np.asarray(parents, dtype=np.int64)

    return expand


def initial_frontier() -> Frontier:
    return Frontier(
        states=readonly(state.initial()),
        contexts=readonly(np.zeros((1, 1), dtype=np.float64)),
        energies=readonly(np.array([float("inf")], dtype=np.float64)),
    )


def exhaustive_best(seed: int, n: int, depth: int, cutoff: float) -> list[float]:
    """Complete enumeration of every reachable state — the ground truth for the lowest energy per depth.

    Not a reimplementation of the search: no width, beams, or queues — just every state whose prefixes
    all survive the cutoff."""
    frontier: dict[tuple[int, ...], float] = {(): 0.0}
    best = []
    for _ in range(depth):
        children = {(*row, j): node_energy(seed, (*row, j)) + energy for row, energy in frontier.items() for j in range(n) if j not in row}
        frontier = {row: energy for row, energy in children.items() if energy <= cutoff}
        if not frontier:
            break
        best.append(min(frontier.values()))
    return best


def greedy_reference(seed: int, n: int, depth: int, cutoff: float) -> list[tuple[tuple[int, ...], float]]:
    """The greedy contract stated directly: repeatedly append the surviving child with the lowest energy."""
    row: tuple[int, ...] = ()
    energy = 0.0
    trajectory = []
    for _ in range(depth):
        children = [((*row, j), node_energy(seed, (*row, j)) + energy) for j in range(n) if j not in row]
        children = [(child, child_energy) for child, child_energy in children if child_energy <= cutoff]
        if not children:
            break
        row, energy = min(children, key=lambda entry: entry[1])
        trajectory.append((row, energy))
    return trajectory


def check_beam(seed: int, n: int, width: int, depth: int, beams: int, cutoff: float) -> None:
    def run(beams: int) -> list[Frontier]:
        S = strategy.beam(accumulating_energy(seed), ordered_forward(n), width=width, depth=depth, beams=beams, cutoff=cutoff)
        return list(S(initial_frontier(), np.random.default_rng(0)))

    trajectory = run(beams)
    best = exhaustive_best(seed, n, depth, cutoff)

    assert len(trajectory) <= len(best)  # the search cannot reach depths complete enumeration cannot
    for d, frontier in enumerate(trajectory, start=1):
        row = tuple(frontier.states[0].tolist())
        energy = float(frontier.energies[0])
        assert len(frontier) == 1
        assert frontier.states.shape == (1, d)
        assert set(row) <= set(range(n)) and len(set(row)) == d  # a valid selection of d distinct indices
        assert energy == path_energy(seed, row)  # pairs the yielded state with its true energy; catches context misrouting
        assert frontier.contexts.tolist() == [[energy]]
        assert energy <= cutoff
        assert energy >= best[d - 1]  # cannot beat complete enumeration

    # With width large enough to hold every reachable state, the search degenerates to complete
    # enumeration and the yields must match it exactly.
    if width >= sum(math.perm(n, d) for d in range(1, depth + 1)):
        assert [float(frontier.energies[0]) for frontier in trajectory] == best

    # An extra beam only adds expansions: it never loses depths and never worsens a yield.
    wider = run(beams + 1)
    assert len(wider) >= len(trajectory)
    for frontier, wider_frontier in zip(trajectory, wider):
        assert float(wider_frontier.energies[0]) <= float(frontier.energies[0])


def check_greedy(seed: int, n: int, depth: int, cutoff: float) -> None:
    S = strategy.greedy(accumulating_energy(seed), ordered_forward(n), depth=depth, cutoff=cutoff)
    trajectory = list(S(initial_frontier(), np.random.default_rng(0)))
    expected = greedy_reference(seed, n, depth, cutoff)

    assert len(trajectory) == len(expected)
    for frontier, (row, energy) in zip(trajectory, expected, strict=True):
        assert len(frontier) == 1
        assert frontier.states.tolist() == [list(row)]
        assert frontier.energies.tolist() == [energy]
        assert frontier.contexts.tolist() == [[energy]]


@st.composite
def beam_problems(draw):
    seed = draw(st.integers(0, 2**32 - 1))
    n = draw(st.integers(0, 5))
    width = draw(st.sampled_from([1, 2, 3, 400]))  # 400 exceeds every reachable-state count for n <= 5: the exhaustive regime
    depth = draw(st.integers(0, n + 2))  # depth may exceed n: the trajectory must truncate when the universe dries up
    beams = draw(st.integers(1, 3))
    cutoff = float("inf") if draw(st.booleans()) else draw(st.floats(-1.5, 1.5))
    return BeamProblem(seed, n, width, depth, beams, cutoff)


@st.composite
def greedy_problems(draw):
    seed = draw(st.integers(0, 2**32 - 1))
    n = draw(st.integers(0, 5))
    depth = draw(st.integers(0, n + 2))
    cutoff = float("inf") if draw(st.booleans()) else draw(st.floats(-1.5, 1.5))
    return GreedyProblem(seed, n, depth, cutoff)


BEAM_VALID = {
    "greedy-equivalent": BeamCase(0, 5, 1, 3, 1, float("inf")),
    "classic-beam": BeamCase(1, 5, 2, 3, 1, float("inf")),
    "multi-beam": BeamCase(2, 5, 2, 3, 3, float("inf")),
    "exhaustive-width": BeamCase(3, 4, 400, 4, 1, float("inf")),
    "zero-depth": BeamCase(3, 5, 2, 0, 1, float("inf")),
    # No indices to select at all — the trajectory is empty.
    "empty-universe": BeamCase(4, 0, 1, 3, 1, float("inf")),
    # Only 2 indices — depths 3..5 are unreachable, so the trajectory stops early.
    "universe-dries-up": BeamCase(5, 2, 2, 5, 3, float("inf")),
    # Node values are U(-1, 1), so a cutoff of 0 prunes roughly half the children.
    "cutoff-prunes-branches": BeamCase(6, 4, 2, 3, 2, 0.0),
    # Every child exceeds the cutoff — nothing survives depth 1.
    "cutoff-drops-everything": BeamCase(7, 3, 2, 2, 1, -2.0),
}

BEAM_INVALID = {
    "zero-width": (BeamCase(0, 3, 0, 1, 1, float("inf")), "width"),
    "negative-depth": (BeamCase(0, 3, 1, -1, 1, float("inf")), "depth"),
    "zero-beams": (BeamCase(0, 3, 1, 1, 0, float("inf")), "beams"),
}

GREEDY_VALID = {
    "cutoff-free": GreedyCase(0, 5, 3, float("inf")),
    "cutoff": GreedyCase(1, 4, 3, 0.0),
}

GREEDY_INVALID = {
    "negative-depth": (GreedyCase(0, 3, -1, float("inf")), "depth"),
}


@given(problem=beam_problems())
def test_beam_compatibility(problem: BeamProblem) -> None:
    check_beam(*problem)


@pytest.mark.parametrize("case", BEAM_VALID.values(), ids=BEAM_VALID.keys())
def test_beam_valid(case: BeamCase) -> None:
    check_beam(*case)


@pytest.mark.parametrize("case,match", BEAM_INVALID.values(), ids=BEAM_INVALID.keys())
def test_beam_invalid(case: BeamCase, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        strategy.beam(
            accumulating_energy(case.seed), ordered_forward(case.n), width=case.width, depth=case.depth, beams=case.beams, cutoff=case.cutoff
        )


def test_beam_worked_example() -> None:
    """Hand-derived anchor for the two behaviors the outward checks cannot pin: the pop width and
    cross-beam incumbent updates.

        depth 1: (0,) -> 0, (1,) -> 1, (2,) -> 2
        depth 2: (0, 1) -> 10, (0, 2) -> 11, (1, 0) -> 8, (1, 2) -> -5

    width=1, beams=1 pops only (0,) and must yield 10 — popping wider would leak -5 in.
    width=1, beams=2: the second beam pops the leftover (1,) and must improve depth 2 to -5.
    width=2, beams=1 pops (0,) and (1,) together and finds -5 in a single pass."""
    TABLE = {(0,): 0.0, (1,): 1.0, (2,): 2.0, (0, 1): 10.0, (0, 2): 11.0, (1, 0): 8.0, (1, 2): -5.0}

    def E(states: States, _contexts: Contexts) -> tuple[Energies, Contexts]:
        return np.array([TABLE[tuple(row)] for row in states.tolist()]), np.empty((states.shape[0], 0), dtype=np.float64)

    def run(width: int, beams: int) -> list[tuple[list[int], float]]:
        S = strategy.beam(E, ordered_forward(3), width=width, depth=2, beams=beams)
        return [(frontier.states[0].tolist(), float(frontier.energies[0])) for frontier in S(initial_frontier(), np.random.default_rng(0))]

    assert run(width=1, beams=1) == [([0], 0.0), ([0, 1], 10.0)]
    assert run(width=1, beams=2) == [([0], 0.0), ([1, 2], -5.0)]
    assert run(width=2, beams=1) == [([0], 0.0), ([1, 2], -5.0)]


@given(problem=greedy_problems())
def test_greedy_compatibility(problem: GreedyProblem) -> None:
    check_greedy(*problem)


@pytest.mark.parametrize("case", GREEDY_VALID.values(), ids=GREEDY_VALID.keys())
def test_greedy_valid(case: GreedyCase) -> None:
    check_greedy(*case)


@pytest.mark.parametrize("case,match", GREEDY_INVALID.values(), ids=GREEDY_INVALID.keys())
def test_greedy_invalid(case: GreedyCase, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        strategy.greedy(accumulating_energy(case.seed), ordered_forward(case.n), depth=case.depth, cutoff=case.cutoff)
