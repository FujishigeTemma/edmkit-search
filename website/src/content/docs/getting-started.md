---
title: Getting Started
description: Install edmkit-search and run your first search.
---

## Installation

Install from PyPI:

```bash
pip install edmkit-search
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add edmkit-search
```

:::note
edmkit-search requires Python 3.13 or later and pulls in [`edmkit`](https://github.com/FujishigeTemma/edmkit) as a dependency.
:::

## Requirements

edmkit-search depends on:

- **edmkit** >= 0.0.9 — embedding, simplex projection, S-Map, metrics, splits
- **NumPy** >= 2.4 — core array operations

## The Mental Model

A search in edmkit-search is a loop over a *frontier* of candidate states. Each iteration:

1. **Expand** the frontier into children via a `Neighborhood`,
2. **Score** the children with an `Energy` to obtain energies and contexts,
3. **Select** the next frontier with a `Strategy`.

Four orthogonal abstractions, plus a `Dataset` that holds the underlying time series:

- **`State`** — `(N, d)` ndarray of indices into the dataset.
- **`Neighborhood`** — `(parents, rng) → (children, parents_idx)`.
- **`Energy`** — `(states, contexts) → (energies, contexts)`, with a deferred-job `Plan` form for parallel execution.
- **`Strategy`** — a `Step` that consumes a `Frontier` and returns the next one.

## Your First Search

The example below recovers the informative columns of a Lorenz-96 trajectory mixed with autoregressive noise. It is the smaller cousin of [`e2e/synthetic.py`](https://github.com/FujishigeTemma/edmkit-search/blob/main/e2e/synthetic.py).

```python
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from edmkit.simplex_projection import simplex_projection
from edmkit.metrics import mean_rho as _mean_rho
from edmkit.splits import temporal_fold

from edmkit.search import energy, neighborhood, state, strategy
from edmkit.search.dataset import Dataset, Subset


# 1. Frame the metric so that lower is better — strategies minimize energy.
def mean_rho(predictions: np.ndarray, observations: np.ndarray) -> np.ndarray:
    return 1.0 - _mean_rho(predictions.reshape(observations.shape), observations)


# 2. Wrap the data, carve a temporal split for the outer evaluation.
data = Dataset(X=X_full, Y=Y_full)
fold1 = temporal_fold(data.X.shape[0], train_ratio=0.8)
train = Subset(data, fold1.train)
validation = Subset(data, fold1.validation)

# 3. Build the energy from an inner split of the training arm.
fold2 = temporal_fold(train.X.shape[0], train_ratio=0.75)
initial_context, plan = energy.holdout(
    data=train,
    fold=fold2,
    predict=simplex_projection,
    metric=mean_rho,
    batch_size=64,
)

# 4. Run the search. The energy wrapper is a closure over `plan` and `pool`,
#    so the library never touches a thread.
with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
    def E(states: state.States, contexts: energy.Contexts) -> tuple[energy.Energies, energy.Contexts]:
        futures = [pool.submit(job) for job in plan(states, contexts)]
        n = states.shape[0]
        energies = np.empty(n, dtype=np.float64)
        new_contexts = np.empty((n, initial_context.shape[1]), dtype=np.float64)
        for f in futures:
            s, e, c = f.result()
            energies[s] = e
            new_contexts[s] = c
        return energies, new_contexts

    N = neighborhood.forward(data.X.shape[1])
    S = strategy.greedy(E, N)
    initial = strategy.Frontier(
        states=state.initial(),
        contexts=initial_context,
        energies=np.array([float("inf")], dtype=np.float64),
    )

    trace = list(strategy.run(initial, S, max_steps=8, rng=np.random.default_rng(0)))

# 5. Score the trajectory on the held-out arm.
predictions = np.zeros((len(trace), *validation.Y.shape))
for j in range(len(trace)):
    selected = trace[j].states[0]
    predictions[j] = simplex_projection(train.X[:, selected], train.Y, validation.X[:, selected]).reshape(validation.Y.shape)
validation_scores = mean_rho(predictions, np.broadcast_to(validation.Y, predictions.shape))

selected = trace[-1].states[0]  # final selected indices
best_step = int(np.argmin(validation_scores))
```

## What's Next?

- Read [The Search Loop](/edmkit-search/concepts/search-loop/) for the conceptual picture.
- Browse the full [API Reference](/edmkit-search/reference/dataset/).
- See [`e2e/synthetic.py`](https://github.com/FujishigeTemma/edmkit-search/blob/main/e2e/synthetic.py) for the full pipeline (data generation, variable filtering, two-level fold split, per-step validation curve).
