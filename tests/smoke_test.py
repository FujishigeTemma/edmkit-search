import numpy as np
from edmkit.metrics import mean_rho
from edmkit.simplex_projection import simplex_projection
from edmkit.splits import temporal_fold

from edmkit.search import energy, neighborhood, state, strategy
from edmkit.search.dataset import Dataset, Subset
from edmkit.search.energy import Contexts, Energies
from edmkit.search.state import States


def corr(predictions: np.ndarray, observations: np.ndarray) -> np.ndarray:
    return 1.0 - mean_rho(predictions.reshape(observations.shape), observations)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 5))
    Y = X[:, :1] + 0.1 * rng.standard_normal((200, 1))

    data = Dataset(X=X, Y=Y)
    train = Subset(data, temporal_fold(len(data), 0.8).train)
    initial_context, plan = energy.cross.holdout(
        data=train,
        fold=temporal_fold(len(train), 0.75),
        predict=simplex_projection,
        metric=corr,
    )

    def E(states: States, contexts: Contexts) -> tuple[Energies, Contexts]:
        energies = np.empty(states.shape[0], dtype=np.float64)
        new_contexts = np.empty((states.shape[0], initial_context.shape[1]), dtype=np.float64)
        for job in plan(states, contexts):
            sl, e, c = job()
            energies[sl], new_contexts[sl] = e, c
        return energies, new_contexts

    S = strategy.greedy(E, neighborhood.forward(data.X.shape[1]), depth=2)
    initial = strategy.Frontier(
        states=state.initial(),
        contexts=initial_context,
        energies=np.array([float("inf")], dtype=np.float64),
    )

    trace = list(S(initial, rng))
    assert len(trace) == 2
    assert all(np.isfinite(frontier.energies).all() for frontier in trace)
