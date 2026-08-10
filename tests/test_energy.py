from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pytest
from edmkit import simplex_projection
from edmkit.metrics import mae
from edmkit.splits import Fold
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import softmax as scipy_softmax

from edmkit.search import energy
from edmkit.search.dataset import Dataset
from edmkit.search.energy import Contexts, Energies, Energy, Plan
from edmkit.search.state import States


class EnergyProblem(NamedTuple):
    data: Dataset
    folds: list[Fold]
    states: States
    contexts: Contexts
    theiler_window: int
    batch_size: int


class SoftmaxProblem(NamedTuple):
    values: np.ndarray
    temperature: float


class SoftmaxCase(NamedTuple):
    values: np.ndarray
    temperature: float


def serial(initial: Contexts, plan: Plan) -> Energy:
    """Drive a Plan serially into an Energy — the minimal executor (e2e runs the same wiring on a thread pool).

    Also enforces the Plan contract: the yielded jobs' slices must cover the whole batch."""

    def E(states: States, contexts: Contexts) -> tuple[Energies, Contexts]:
        energies = np.empty(states.shape[0], dtype=np.float64)
        new_contexts = np.empty((states.shape[0], initial.shape[1]), dtype=np.float64)
        covered = np.zeros(states.shape[0], dtype=bool)
        for job in plan(states, contexts):
            sl, e, c = job()
            energies[sl], new_contexts[sl], covered[sl] = e, c, True
        assert covered.all(), "plan's jobs must cover every state"
        return energies, new_contexts

    return E


def colsum_predict(X: np.ndarray, Y: np.ndarray, Q: np.ndarray, *, mask: np.ndarray | None = None) -> np.ndarray:
    """Deterministic PredictFunc stand-in: each prediction is the query's coordinate sum, repeated per target.

    Shape-generic over the 2D (reference) and 3D (batched plan) calls, so both sides see the same predictor."""
    return Q.sum(axis=-1, keepdims=True).repeat(Y.shape[-1], axis=-1)


def fold_loss(data: Dataset, fold: Fold, row: Sequence[int], *, within: bool) -> float:
    """Single-state, single-fold loss: 2D arrays, no batching, no transposes."""
    columns = list(row)
    X, Q = data.X[fold.train][:, columns], data.X[fold.validation][:, columns]
    Y = data.Y[fold.train][:, columns] if within else data.Y[fold.train]
    observed = data.Y[fold.validation][:, columns] if within else data.Y[fold.validation]
    return float(mae(colsum_predict(X, Y, Q), observed))


def holdout_reference(data: Dataset, fold: Fold, states: States, *, within: bool) -> list[float]:
    return [fold_loss(data, fold, row, within=within) for row in states.tolist()]


def folds_reference(data: Dataset, folds: Sequence[Fold], states: States, *, within: bool) -> np.ndarray:
    return np.array([[fold_loss(data, fold, row, within=within) for fold in folds] for row in states.tolist()])


def loo_reference(data: Dataset, states: States, theiler_window: int, *, within: bool) -> list[float]:
    """Per-state LOO losses via batches of one — pins the batched plan to the unbatched call."""
    losses = []
    for row in states.tolist():
        columns = list(row)
        X = data.X[:, columns][None]
        Y = (data.Y[:, columns] if within else data.Y)[None]
        losses.append(float(mae(simplex_projection.loo(X, Y, theiler_window=theiler_window), Y)[0]))
    return losses


def softmax_reference(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """scipy's softmax as an independent oracle — stable under extreme logits too."""
    return scipy_softmax(np.asarray(values, dtype=np.float64) / temperature, axis=1)


def check_holdout(problem: EnergyProblem, holdout, *, within: bool) -> None:
    fold = problem.folds[0]
    initial, plan = holdout(data=problem.data, fold=fold, predict=colsum_predict, metric=mae, batch_size=problem.batch_size)
    assert initial.shape == (1, 0)

    energies, contexts = serial(initial, plan)(problem.states, np.empty((problem.states.shape[0], 0), dtype=np.float64))

    np.testing.assert_allclose(energies, holdout_reference(problem.data, fold, problem.states, within=within), rtol=1e-5, atol=1e-6)
    assert contexts.shape == (problem.states.shape[0], 0)


def check_folds(problem: EnergyProblem, folds, *, within: bool) -> None:
    initial, plan = folds(
        data=problem.data,
        folds=problem.folds,
        predict=colsum_predict,
        metric=mae,
        weight=energy.softmax(),
        batch_size=problem.batch_size,
    )
    np.testing.assert_array_equal(initial, np.zeros((1, len(problem.folds))))

    energies, contexts = serial(initial, plan)(problem.states, problem.contexts)

    losses = folds_reference(problem.data, problem.folds, problem.states, within=within)
    np.testing.assert_allclose(contexts, losses, rtol=1e-5, atol=1e-6)
    # The energy is the weighted stepwise delta: weights come from the *incoming* context, losses replace it.
    expected = (softmax_reference(problem.contexts) * (losses - problem.contexts)).sum(axis=1)
    np.testing.assert_allclose(energies, expected, rtol=1e-5, atol=1e-6)


def check_loo(problem: EnergyProblem, loo, *, within: bool) -> None:
    initial, plan = loo(data=problem.data, metric=mae, theiler_window=problem.theiler_window, batch_size=problem.batch_size)
    assert initial.shape == (1, 0)

    energies, contexts = serial(initial, plan)(problem.states, np.empty((problem.states.shape[0], 0), dtype=np.float64))

    np.testing.assert_allclose(energies, loo_reference(problem.data, problem.states, problem.theiler_window, within=within), rtol=1e-5, atol=1e-6)
    assert contexts.shape == (problem.states.shape[0], 0)


def check_softmax(values: np.ndarray, temperature: float) -> None:
    np.testing.assert_allclose(energy.softmax(temperature)(values), softmax_reference(values, temperature), rtol=1e-12, atol=1e-15)


@st.composite
def energy_problems(draw):
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    T = draw(st.integers(12, 24))
    D = draw(st.integers(2, 5))
    # Y is full-width so the same dataset serves both cross (fixed target) and within (per-state target) plans.
    data = Dataset(rng.normal(size=(T, D)), rng.normal(size=(T, D)))

    folds = []
    for _ in range(draw(st.integers(1, 3))):
        # Arbitrary index sets: plans must not assume contiguous, ordered, or exhaustive folds.
        order = rng.permutation(T)
        split = draw(st.integers(4, T - 4))
        folds.append(Fold(train=order[:split], validation=order[split:]))
    d = draw(st.integers(1, min(3, D)))
    states = np.stack([rng.permutation(D)[:d] for _ in range(draw(st.integers(1, 4)))]).astype(np.int64)
    contexts = rng.normal(size=(states.shape[0], len(folds)))
    theiler_window = draw(st.integers(0, 2))
    batch_size = draw(st.sampled_from([1, 2, 10000]))
    return EnergyProblem(data, folds, states, contexts, theiler_window, batch_size)


@st.composite
def softmax_problems(draw):
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    values = rng.uniform(-10.0, 10.0, size=(draw(st.integers(1, 4)), draw(st.integers(1, 5))))
    return SoftmaxProblem(values, draw(st.floats(0.1, 10.0)))


HOLDOUT_MODES = {"cross": (energy.cross.holdout, False), "within": (energy.within.holdout, True)}
FOLDS_MODES = {"cross": (energy.cross.folds, False), "within": (energy.within.folds, True)}
LOO_MODES = {"cross": (energy.cross.loo, False), "within": (energy.within.loo, True)}

DATA = Dataset(np.arange(18.0).reshape(6, 3), np.arange(18.0).reshape(6, 3))

SOFTMAX_VALID = {
    "uniform-on-equal-logits": SoftmaxCase(np.zeros((2, 4)), 1.0),
    # A naive exp would overflow here; both the implementation and the scipy oracle must stay stable.
    "extreme-logits": SoftmaxCase(np.array([[1e6, 0.0, -1e6], [-1e6, -1e6, -1e6]]), 1.0),
}

SOFTMAX_INVALID = {
    "zero-temperature": 0.0,
    "negative-temperature": -1.0,
}


@pytest.mark.parametrize("holdout,within", HOLDOUT_MODES.values(), ids=HOLDOUT_MODES.keys())
@given(problem=energy_problems())
def test_holdout_compatibility(holdout, within: bool, problem: EnergyProblem) -> None:
    check_holdout(problem, holdout, within=within)


@pytest.mark.parametrize("folds,within", FOLDS_MODES.values(), ids=FOLDS_MODES.keys())
@given(problem=energy_problems())
def test_folds_compatibility(folds, within: bool, problem: EnergyProblem) -> None:
    check_folds(problem, folds, within=within)


@pytest.mark.parametrize("folds", [builder for builder, _ in FOLDS_MODES.values()], ids=FOLDS_MODES.keys())
def test_folds_invalid(folds) -> None:
    with pytest.raises(ValueError, match="folds"):
        folds(data=DATA, folds=[], predict=colsum_predict, metric=mae, weight=energy.softmax())


@pytest.mark.parametrize("loo,within", LOO_MODES.values(), ids=LOO_MODES.keys())
@settings(deadline=None)
@given(problem=energy_problems())
def test_loo_compatibility(loo, within: bool, problem: EnergyProblem) -> None:
    check_loo(problem, loo, within=within)


@pytest.mark.parametrize("loo", [builder for builder, _ in LOO_MODES.values()], ids=LOO_MODES.keys())
def test_loo_invalid(loo) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        loo(data=DATA, metric=mae, theiler_window=-1)


@given(problem=softmax_problems())
def test_softmax_compatibility(problem: SoftmaxProblem) -> None:
    check_softmax(*problem)


@pytest.mark.parametrize("case", SOFTMAX_VALID.values(), ids=SOFTMAX_VALID.keys())
def test_softmax_valid(case: SoftmaxCase) -> None:
    check_softmax(*case)


@pytest.mark.parametrize("temperature", SOFTMAX_INVALID.values(), ids=SOFTMAX_INVALID.keys())
def test_softmax_invalid(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        energy.softmax(temperature=temperature)
