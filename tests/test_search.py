from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence

import numpy as np
import pytest
from edmkit.metrics import mae as metric_mae
from edmkit.splits import Fold

from edmkit.search import energy, neighborhood, state, strategy
from edmkit.search.dataset import Dataset
from edmkit.search.energy import Contexts, Energies, Energy, Plan
from edmkit.search.neighborhood import Neighborhood
from edmkit.search.state import States


def colsum_predict(
    X: np.ndarray,
    Y: np.ndarray,
    Q: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    if Q.ndim == 2 and Y.ndim == 2:
        return Q.sum(axis=1, keepdims=True).repeat(Y.shape[1], axis=1)
    if Q.ndim == 3 and Y.ndim == 3:
        return Q.sum(axis=2, keepdims=True).repeat(Y.shape[2], axis=2)
    raise ValueError(f"expected 2D or 3D predict arrays, got Q={Q.ndim}D, Y={Y.ndim}D")


def arrays() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 1.0, 0.0],
            [3.0, 2.0, 1.0],
            [4.0, 3.0, 2.0],
            [5.0, 5.0, 1.0],
        ],
        dtype=np.float64,
    )
    Y = np.array([[0.5], [1.5], [1.0], [2.5], [3.0], [4.0]], dtype=np.float64)
    return X, Y


def folds_for_arrays() -> list[Fold]:
    return [
        Fold(train=np.array([0, 1, 2]), validation=np.array([3, 4])),
        Fold(train=np.array([2, 3, 4]), validation=np.array([0, 5])),
    ]


def uniform(values: np.ndarray) -> np.ndarray:
    return np.full(values.shape, 1.0 / values.shape[-1], dtype=np.float64)


def per_fold_loss(
    X: np.ndarray,
    Y: np.ndarray,
    folds: Sequence[Fold],
    indices: Sequence[int],
) -> np.ndarray:
    columns = list(indices)
    out = np.empty(len(folds), dtype=np.float64)
    for f, fold in enumerate(folds):
        prediction = colsum_predict(
            X[fold.train][:, columns], Y[fold.train], X[fold.validation][:, columns]
        )
        out[f] = float(metric_mae(prediction, Y[fold.validation]))
    return out


# -------------------------------------------------------------------- helpers


def to_energy(initial: Contexts, plan: Plan) -> Energy:
    """Drive a Plan serially into an Energy. Matches the inline loop used in callers."""
    c_dim = initial.shape[1]

    def E(states: States, contexts: Contexts) -> tuple[Energies, Contexts]:
        n = states.shape[0]
        energies = np.empty(n, dtype=np.float64)
        new_contexts = np.empty((n, c_dim), dtype=np.float64)
        for job in plan(states, contexts):
            sl, e, c = job()
            energies[sl] = e
            new_contexts[sl] = c
        return energies, new_contexts

    return E


def constant_energy(fn: Callable[[States], np.ndarray]) -> tuple[Contexts, Energy]:
    """Energy with empty per-state contexts; energies come from `fn(states)`."""
    initial = np.empty((1, 0), dtype=np.float64)

    def E(states: States, _contexts: Contexts) -> tuple[Energies, Contexts]:
        energies = np.asarray(fn(states), dtype=np.float64)
        return energies, np.empty((states.shape[0], 0), dtype=np.float64)

    return initial, E


def empty_neighborhood() -> Neighborhood:
    """Neighborhood that always produces no children."""

    def expand(states: States, _rng: np.random.Generator) -> tuple[States, np.ndarray]:
        return (
            np.empty((0, states.shape[1] + 1), dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    return expand


def fixed_neighborhood(perturbations: Sequence[int]) -> Neighborhood:
    """Append each element of `perturbations` to every parent, preserving order."""
    ps = np.asarray(perturbations, dtype=np.int64)

    def expand(states: States, _rng: np.random.Generator) -> tuple[States, np.ndarray]:
        N = states.shape[0]
        prefix = np.repeat(states, len(ps), axis=0)
        suffix = np.tile(ps, N)[:, None]
        children = np.concatenate([prefix, suffix], axis=1)
        parents = np.repeat(np.arange(N, dtype=np.int64), len(ps))
        return children, parents

    return expand


def initial_frontier(initial: Contexts) -> strategy.Frontier:
    """Single-state frontier from the empty initial state."""
    return strategy.Frontier(
        states=state.initial(),
        contexts=initial,
        energies=np.array([float("inf")], dtype=np.float64),
    )


# ---------------------------------------------------------------- neighborhood


class TestForward:
    def test_children_extend_parents_with_unselected_indices(self) -> None:
        N = neighborhood.forward(6)
        states = np.array([[1, 4]], dtype=np.int64)
        children, parents = N(states, np.random.default_rng(0))

        assert children.shape == (4, 3)
        assert parents.shape == (4,)
        np.testing.assert_array_equal(children[:, :2], np.broadcast_to([1, 4], (4, 2)))
        assert set(children[:, 2].tolist()) == {0, 2, 3, 5}
        np.testing.assert_array_equal(parents, np.zeros(4, dtype=np.int64))

    def test_each_parent_expands_independently(self) -> None:
        N = neighborhood.forward(4)
        states = np.array([[0], [3]], dtype=np.int64)
        children, parents = N(states, np.random.default_rng(0))

        assert children.shape == (6, 2)
        np.testing.assert_array_equal(parents, np.array([0, 0, 0, 1, 1, 1]))
        assert set(children[:3, 1].tolist()) == {1, 2, 3}
        assert set(children[3:, 1].tolist()) == {0, 1, 2}

    def test_same_seed_gives_same_children_order(self) -> None:
        N = neighborhood.forward(8)
        states = np.array([[3]], dtype=np.int64)

        c1, p1 = N(states, np.random.default_rng(42))
        c2, p2 = N(states, np.random.default_rng(42))

        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(p1, p2)

    def test_full_state_yields_no_children(self) -> None:
        N = neighborhood.forward(3)
        states = np.array([[0, 1, 2]], dtype=np.int64)
        children, parents = N(states, np.random.default_rng(0))

        assert children.shape == (0, 4)
        assert parents.shape == (0,)

    def test_empty_batch_yields_no_children(self) -> None:
        N = neighborhood.forward(5)
        states = np.empty((0, 2), dtype=np.int64)
        children, parents = N(states, np.random.default_rng(0))

        assert children.shape == (0, 3)
        assert parents.shape == (0,)

    def test_negative_n_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            neighborhood.forward(-1)


# ------------------------------------------------------------------- strategy


class TestStrategy:
    def test_greedy_equals_beam_width_one(self) -> None:
        initial, E = constant_energy(lambda states: states.sum(axis=1).astype(np.float64))
        N = neighborhood.forward(5)

        greedy_step = strategy.greedy(E, N)
        beam_step = strategy.beam(E, N, width=1)

        greedy_trace = [
            f.states[0].copy()
            for f in strategy.run(
                initial_frontier(initial),
                greedy_step,
                max_steps=3,
                rng=np.random.default_rng(7),
            )
        ]
        beam_trace = [
            f.states[0].copy()
            for f in strategy.run(
                initial_frontier(initial),
                beam_step,
                max_steps=3,
                rng=np.random.default_rng(7),
            )
        ]

        assert len(greedy_trace) == len(beam_trace) == 3
        for g, b in zip(greedy_trace, beam_trace, strict=True):
            np.testing.assert_array_equal(g, b)

    def test_run_stops_on_empty_frontier(self) -> None:
        initial, E = constant_energy(lambda states: np.zeros(states.shape[0]))
        trace = list(
            strategy.run(
                initial_frontier(initial),
                strategy.greedy(E, empty_neighborhood()),
                max_steps=3,
                rng=np.random.default_rng(0),
            )
        )

        assert trace == []

    def test_run_respects_max_steps(self) -> None:
        initial, E = constant_energy(
            lambda states: np.full(states.shape[0], states.shape[1])
        )
        trace = list(
            strategy.run(
                initial_frontier(initial),
                strategy.greedy(E, neighborhood.forward(5)),
                max_steps=2,
                rng=np.random.default_rng(0),
            )
        )

        assert len(trace) == 2

    def test_cutoff_filters_after_energy_evaluation(self) -> None:
        seen_indices: list[int] = []

        def fn(states: States) -> np.ndarray:
            seen_indices.extend(states[:, -1].tolist())
            return states[:, -1].astype(np.float64)

        initial, E = constant_energy(fn)
        N = fixed_neighborhood([0, 1, 2])

        step = strategy.beam(E, N, width=3, cutoff=1.0)
        out = step(initial_frontier(initial), np.random.default_rng(0))

        assert seen_indices == [0, 1, 2]
        assert out.states[:, -1].tolist() == [0, 1]

    def test_frontier_states_expand_independently(self) -> None:
        _, E = constant_energy(lambda states: states[:, -1].astype(np.float64))
        step = strategy.beam(E, neighborhood.forward(3), width=10)
        frontier = strategy.Frontier(
            states=np.array([[0], [1]], dtype=np.int64),
            contexts=np.empty((2, 0), dtype=np.float64),
            energies=np.zeros(2, dtype=np.float64),
        )

        out = step(frontier, np.random.default_rng(1))

        assert out.states.shape[1] == 2
        rows = {tuple(row) for row in out.states.tolist()}
        assert (0, 0) not in rows
        assert (1, 1) not in rows

    def test_beam_width_must_be_positive(self) -> None:
        _, E = constant_energy(lambda states: np.zeros(states.shape[0]))
        with pytest.raises(ValueError, match="width"):
            strategy.beam(E, neighborhood.forward(2), width=0)


# --------------------------------------------------------------------- holdout


class TestHoldout:
    def test_step_matches_naive_reference(self) -> None:
        X, Y = arrays()
        X_train, X_val = X[:4], X[4:]
        Y_train, Y_val = Y[:4], Y[4:]
        states = np.array([[0, 1], [1, 2]], dtype=np.int64)

        initial, plan = energy.holdout(
            data=Dataset(X, Y),
            fold=Fold(train=np.arange(4), validation=np.arange(4, 6)),
            predict=colsum_predict,
            metric=metric_mae,
        )
        E = to_energy(initial, plan)

        expected = [
            float(
                metric_mae(
                    colsum_predict(X_train[:, list(row)], Y_train, X_val[:, list(row)]),
                    Y_val,
                )
            )
            for row in states
        ]
        energies, new_contexts = E(states, np.empty((2, 0), dtype=np.float64))

        np.testing.assert_allclose(energies, expected)
        assert new_contexts.shape == (2, 0)

    def test_initial_is_empty_context(self) -> None:
        X, Y = arrays()
        initial, _ = energy.holdout(
            data=Dataset(X, Y),
            fold=Fold(train=np.arange(4), validation=np.arange(4, 6)),
            predict=colsum_predict,
            metric=metric_mae,
        )

        assert initial.shape == (1, 0)


# ----------------------------------------------------------------------- folds


class TestFolds:
    def test_initial_is_zero_baseline(self) -> None:
        X, Y = arrays()
        initial, _ = energy.folds(
            data=Dataset(X, Y),
            folds=folds_for_arrays(),
            predict=colsum_predict,
            metric=metric_mae,
            weight=uniform,
        )

        np.testing.assert_array_equal(initial, np.zeros((1, 2)))

    def test_first_step_with_zeros_context_equals_mean_loss(self) -> None:
        X, Y = arrays()
        split = folds_for_arrays()
        initial, plan = energy.folds(
            data=Dataset(X, Y),
            folds=split,
            predict=colsum_predict,
            metric=metric_mae,
            weight=uniform,
        )
        E = to_energy(initial, plan)
        states = np.array([[0], [1], [2]], dtype=np.int64)
        contexts = np.zeros((3, len(split)), dtype=np.float64)

        energies, new_contexts = E(states, contexts)

        expected = [
            float(np.mean(per_fold_loss(X, Y, split, list(row)))) for row in states
        ]
        np.testing.assert_allclose(energies, expected)
        assert new_contexts.shape == (3, len(split))
        for ctx, row in zip(new_contexts, states, strict=True):
            np.testing.assert_allclose(ctx, per_fold_loss(X, Y, split, list(row)))

    def test_two_step_delta_matches_naive_reference(self) -> None:
        X, Y = arrays()
        split = folds_for_arrays()
        initial, plan = energy.folds(
            data=Dataset(X, Y),
            folds=split,
            predict=colsum_predict,
            metric=metric_mae,
            weight=uniform,
        )
        E = to_energy(initial, plan)

        first = np.array([[0], [1]], dtype=np.int64)
        _, ctxs = E(first, np.zeros((2, len(split)), dtype=np.float64))

        second = np.array([[0, 2], [1, 2]], dtype=np.int64)
        energies, _ = E(second, ctxs)

        expected = []
        for parent_loss, child in zip(ctxs, second, strict=True):
            child_loss = per_fold_loss(X, Y, split, list(child))
            expected.append(float(uniform(parent_loss) @ (child_loss - parent_loss)))

        np.testing.assert_allclose(energies, expected)

    def test_mixed_contexts_in_one_batch_evaluate_independently(self) -> None:
        X, Y = arrays()
        split = folds_for_arrays()
        initial, plan = energy.folds(
            data=Dataset(X, Y),
            folds=split,
            predict=colsum_predict,
            metric=metric_mae,
            weight=uniform,
        )
        E = to_energy(initial, plan)

        states = np.array([[0, 1], [1, 2]], dtype=np.int64)
        contexts = np.array(
            [[0.5, 0.5], [0.0, 1.0]],
            dtype=np.float64,
        )

        energies, _ = E(states, contexts)

        loss_a = per_fold_loss(X, Y, split, [0, 1])
        loss_b = per_fold_loss(X, Y, split, [1, 2])
        expected = [
            float(uniform(contexts[0]) @ (loss_a - contexts[0])),
            float(uniform(contexts[1]) @ (loss_b - contexts[1])),
        ]
        np.testing.assert_allclose(energies, expected)

    def test_empty_folds_raises(self) -> None:
        X, Y = arrays()
        with pytest.raises(ValueError, match="folds"):
            energy.folds(
                data=Dataset(X, Y),
                folds=[],
                predict=colsum_predict,
                metric=metric_mae,
                weight=uniform,
            )


# ------------------------------------------------------------------------- loo


class TestLoo:
    def test_step_passes_theiler_window_to_simplex(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loo_module = importlib.import_module("edmkit.search.energy.loo")
        calls: list[int] = []

        def fake_loo(
            X: np.ndarray, Y: np.ndarray, *, theiler_window: int
        ) -> np.ndarray:
            calls.append(theiler_window)
            return np.zeros_like(Y)

        monkeypatch.setattr(loo_module.simplex_projection, "loo", fake_loo)
        X, Y = arrays()
        initial, plan = loo_module.loo(
            data=Dataset(X, Y), metric=metric_mae, theiler_window=6
        )
        E = to_energy(initial, plan)

        E(np.array([[0, 1, 2]], dtype=np.int64), np.empty((1, 0), dtype=np.float64))

        assert calls == [6]

    def test_negative_theiler_window_raises(self) -> None:
        X, Y = arrays()
        with pytest.raises(ValueError, match="non-negative"):
            energy.loo(data=Dataset(X, Y), metric=metric_mae, theiler_window=-1)


# ------------------------------------------------------------------ trajectory


class TestTrajectory:
    """Plan §7 — context flows transparently through the trajectory."""

    def test_context_flows_through_beam_in_lockstep_with_parents(self) -> None:
        X, Y = arrays()
        split = folds_for_arrays()
        initial, plan = energy.folds(
            data=Dataset(X, Y),
            folds=split,
            predict=colsum_predict,
            metric=metric_mae,
            weight=uniform,
        )
        base = to_energy(initial, plan)

        seen: list[np.ndarray] = []

        def traced(states: States, contexts: Contexts) -> tuple[Energies, Contexts]:
            seen.append(contexts.copy())
            return base(states, contexts)

        forward_N = neighborhood.forward(X.shape[1])
        captured_parents: list[np.ndarray] = []

        def N(states: States, rng: np.random.Generator) -> tuple[States, np.ndarray]:
            children, parents = forward_N(states, rng)
            captured_parents.append(parents.copy())
            return children, parents

        beam_step = strategy.beam(traced, N, width=2)
        rng = np.random.default_rng(0)

        first = beam_step(initial_frontier(initial), rng)
        # E in step 1 received initial indexed by the parent map.
        np.testing.assert_array_equal(seen[0], initial[captured_parents[0]])

        beam_step(first, rng)
        # E in step 2 received first.contexts indexed by the parent map.
        np.testing.assert_array_equal(seen[1], first.contexts[captured_parents[1]])

    def test_run_completes_with_readonly_arrays(self) -> None:
        X, Y = arrays()
        split = folds_for_arrays()
        base_initial, base_plan = energy.folds(
            data=Dataset(X, Y),
            folds=split,
            predict=colsum_predict,
            metric=metric_mae,
            weight=uniform,
        )
        base = to_energy(base_initial, base_plan)

        def lock(arr: np.ndarray) -> np.ndarray:
            arr.flags.writeable = False
            return arr

        initial = lock(base_initial.copy())

        def E(states: States, contexts: Contexts) -> tuple[Energies, Contexts]:
            energies, new_contexts = base(states, contexts)
            return energies, lock(new_contexts)

        N = neighborhood.forward(X.shape[1])

        trace = list(
            strategy.run(
                initial_frontier(initial),
                strategy.greedy(E, N),
                max_steps=2,
                rng=np.random.default_rng(0),
            )
        )

        assert len(trace) == 2

    def test_step_is_deterministic(self) -> None:
        X, Y = arrays()
        initial, plan = energy.holdout(
            data=Dataset(X, Y),
            fold=Fold(train=np.arange(4), validation=np.arange(4, 6)),
            predict=colsum_predict,
            metric=metric_mae,
        )
        E = to_energy(initial, plan)
        states = np.array([[0, 1], [1, 2]], dtype=np.int64)
        contexts = np.empty((2, 0), dtype=np.float64)

        e1, _ = E(states, contexts)
        e2, _ = E(states, contexts)

        np.testing.assert_array_equal(e1, e2)


# ---------------------------------------------------------------------- weight


class TestWeight:
    def test_softmax(self) -> None:
        np.testing.assert_allclose(
            energy.softmax()(np.zeros((2, 4))), np.full((2, 4), 0.25)
        )
        assert energy.softmax(temperature=0.01)(np.array([[0.1, 0.8, 0.2]]))[0, 1] > 0.99

    def test_temperature_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            energy.softmax(temperature=0)
