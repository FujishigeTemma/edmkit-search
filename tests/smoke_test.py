"""Smoke tests for importability and one minimal success path per public surface."""

import numpy as np
from edmkit.metrics import mean_rho
from edmkit.simplex_projection import simplex_projection
from edmkit.splits import Fold, temporal_fold

from edmkit.search import energy, neighborhood, state, strategy
from edmkit.search.dataset import Dataset, Subset
from edmkit.search.energy import Contexts, Energies, Energy, Plan
from edmkit.search.state import States


def corr(predictions: np.ndarray, observations: np.ndarray) -> np.ndarray:
    return 1.0 - mean_rho(predictions.reshape(observations.shape), observations)


def to_energy(initial: Contexts, plan: Plan) -> Energy:
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


def test_minimal_greedy_run():
    rng = np.random.default_rng(0)
    T, K = 200, 5
    X = rng.standard_normal((T, K))
    Y = X[:, :1] + 0.1 * rng.standard_normal((T, 1))

    data = Dataset(X=X, Y=Y)
    outer = temporal_fold(len(data), 0.8)
    train = Subset(data, outer.train)
    inner = temporal_fold(len(train), 0.75)

    initial_context, plan = energy.cross.holdout(
        data=train,
        fold=Fold(train=inner.train, validation=inner.validation),
        predict=simplex_projection,
        metric=corr,
    )
    E = to_energy(initial_context, plan)
    N = neighborhood.forward(data.X.shape[1])
    step = strategy.greedy(E, N)
    initial = strategy.Frontier(
        states=state.initial(),
        contexts=initial_context,
        energies=np.array([float("inf")], dtype=np.float64),
    )

    trace = list(strategy.run(initial, step, max_steps=2, rng=rng))
    assert len(trace) == 2
    assert all(np.isfinite(frontier.energies).all() for frontier in trace)


if __name__ == "__main__":
    test_minimal_greedy_run()
