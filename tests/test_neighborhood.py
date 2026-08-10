from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from edmkit.search import neighborhood, state
from edmkit.search.state import States


class ForwardProblem(NamedTuple):
    n: int
    states: States
    seed: int


class ForwardCase(NamedTuple):
    n: int
    states: States


def forward_reference(n: int, states: States) -> list[set[tuple[int, ...]]]:
    """Per parent, the expected children as a set: the parent's indices plus one not-yet-selected index."""
    return [{(*row, j) for j in range(n) if j not in row} for row in states.tolist()]


def check_forward(n: int, states: States, seed: int = 0) -> None:
    N = neighborhood.forward(n)
    children, parents = N(states, np.random.default_rng(seed))
    repeated_children, repeated_parents = N(states, np.random.default_rng(seed))
    per_parent = n - states.shape[1]

    assert children.shape == (states.shape[0] * per_parent, states.shape[1] + 1)
    assert children.dtype == np.int64
    np.testing.assert_array_equal(parents, np.repeat(np.arange(states.shape[0]), per_parent))
    # The per-parent child order is randomized, so compare each parent's block as a set...
    for parent, expected in enumerate(forward_reference(n, states)):
        assert {tuple(row) for row in children[parents == parent].tolist()} == expected
    # ...but the whole expansion must be reproducible from the rng seed.
    np.testing.assert_array_equal(children, repeated_children)
    np.testing.assert_array_equal(parents, repeated_parents)


@st.composite
def forward_problems(draw):
    n = draw(st.integers(0, 8))
    d = draw(st.integers(0, n))
    N = draw(st.integers(0, 4))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    states = np.stack([rng.permutation(n)[:d] for _ in range(N)]) if N else np.empty((0, d), dtype=np.int64)
    return ForwardProblem(n, states, draw(st.integers(0, 2**32 - 1)))


FORWARD_VALID = {
    "single-parent": ForwardCase(6, np.array([[1, 4]], dtype=np.int64)),
    "parents-expand-independently": ForwardCase(4, np.array([[0], [3]], dtype=np.int64)),
    "initial-empty-state": ForwardCase(3, state.initial()),
    "full-state-no-children": ForwardCase(3, np.array([[0, 1, 2]], dtype=np.int64)),
    "empty-batch": ForwardCase(5, np.empty((0, 2), dtype=np.int64)),
    "zero-universe": ForwardCase(0, state.initial()),
}

FORWARD_INVALID = {
    "negative-n": -1,
}


@given(problem=forward_problems())
def test_forward_compatibility(problem: ForwardProblem) -> None:
    check_forward(*problem)


@pytest.mark.parametrize("case", FORWARD_VALID.values(), ids=FORWARD_VALID.keys())
def test_forward_valid(case: ForwardCase) -> None:
    check_forward(*case)


@pytest.mark.parametrize("n", FORWARD_INVALID.values(), ids=FORWARD_INVALID.keys())
def test_forward_invalid(n: int) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        neighborhood.forward(n)
