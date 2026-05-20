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

The example below recovers the informative columns of a Lorenz-96 trajectory that has been mixed with autoregressive noise. It is the smaller cousin of [`e2e/synthetic.py`](https://github.com/FujishigeTemma/edmkit-search/blob/main/e2e/synthetic.py) — read that script for the full pipeline including outer/inner folds and variable filtering.

```python
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from edmkit.simplex_projection import simplex_projection
from edmkit.metrics import mean_rho as _mean_rho
from edmkit.splits import temporal_fold

from edmkit.search import energy, neighborhood, state, strategy
from edmkit.search.dataset import Dataset, Subset
from edmkit.search.energy import Contexts, Energies, Energy, Plan
from edmkit.search.state import States


# 1. Frame metric so that lower is better (this is the "energy" convention).
def mean_rho(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    return 1.0 - _mean_rho(predicted.reshape(observed.shape), observed)


# 2. Wrap a Plan into an Energy by executing jobs on a thread pool.
def parallel(initial: Contexts, plan: Plan, pool: ThreadPoolExecutor) -> Energy:
    def E(states: States, contexts: Contexts) -> tuple[Energies, Contexts]:
        futures = [pool.submit(job) for job in plan(states, contexts)]
        n = states.shape[0]
        energies = np.empty(n, dtype=np.float64)
        new_contexts = np.empty((n, initial.shape[1]), dtype=np.float64)
        for f in futures:
            s, e, c = f.result()
            energies[s] = e
            new_contexts[s] = c
        return energies, new_contexts

    return E


# 3. Build the four pieces.
data = Dataset(X=X_full, Y=Y_full)
outer = temporal_fold(len(data), train_ratio=0.8)
train, validation = Subset(data, outer.train), Subset(data, outer.validation)
inner = temporal_fold(len(train), train_ratio=0.75)

initial_ctx, plan = energy.holdout(
    data=train,
    fold=inner,
    predict=simplex_projection,
    metric=mean_rho,
)

with ThreadPoolExecutor() as pool:
    E = parallel(initial_ctx, plan, pool)
    N = neighborhood.forward(data.X.shape[1])
    step = strategy.greedy(E, N)
    initial = strategy.Frontier(
        states=state.initial(),
        contexts=initial_ctx,
        energies=np.array([float("inf")], dtype=np.float64),
    )

    trace = list(strategy.run(initial, step, max_steps=8, rng=np.random.default_rng(0)))

selected = trace[-1].states[0]            # final selected indices
trajectory = [f.energies[0] for f in trace]  # per-step best energy
```

## What's Next?

- Read [The Search Loop](/edmkit-search/concepts/search-loop/) for the conceptual picture.
- Browse the full [API Reference](/edmkit-search/reference/dataset/).
- See [`e2e/synthetic.py`](https://github.com/FujishigeTemma/edmkit-search/blob/main/e2e/synthetic.py) for the full pipeline (outer/inner folds, parallel filtering, validation scoring).
