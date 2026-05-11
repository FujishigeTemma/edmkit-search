"""Smoke tests for importability and one minimal success path per public surface."""

import numpy as np
from edmkit.metrics import mean_rho as _mean_rho
from edmkit.simplex_projection import simplex_projection
from edmkit.splits import Fold, temporal_fold

from edmkit.search import energy, neighborhood, state, strategy
from edmkit.search.dataset import Dataset, Subset


def mean_rho(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    return 1.0 - _mean_rho(predicted.reshape(observed.shape), observed)


def test_minimal_greedy_run():
    rng = np.random.default_rng(0)
    T, K = 200, 5
    X = rng.standard_normal((T, K))
    Y = X[:, :1] + 0.1 * rng.standard_normal((T, 1))

    data = Dataset(X=X, Y=Y)
    outer = temporal_fold(len(data), train_ratio=0.8)
    train = Subset(data, outer.train)
    inner = temporal_fold(len(train), train_ratio=0.75)

    E = energy.holdout(
        data=train,
        fold=Fold(train=inner.train, validation=inner.validation),
        predict=simplex_projection,
        metric=mean_rho,
    )
    N = neighborhood.forward(data.X.shape[1])
    step = strategy.greedy(E, N)
    initial = strategy.Frontier(
        states=state.initial(),
        contexts=E.initial(),
        energies=np.array([float("inf")], dtype=np.float64),
    )

    trace = list(strategy.run(initial, step, max_steps=2, rng=rng))
    assert len(trace) == 2
    assert all(np.isfinite(frontier.energies).all() for frontier in trace)


if __name__ == "__main__":
    test_minimal_greedy_run()
