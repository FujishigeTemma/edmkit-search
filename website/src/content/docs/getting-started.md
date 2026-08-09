---
title: Getting Started
description: Build a complete variable-selection experiment from scratch in seven steps.
---

This tutorial builds a complete experiment end to end. Starting from a multivariate time series, you run a forward-selection search guided by cross-validated forecast skill, and read off the selected variables and a diagnostic curve.

The example uses [`e2e/synthetic.py`](https://github.com/FujishigeTemma/edmkit-search/blob/main/e2e/synthetic.py): a Lorenz-96 trajectory of `K_signal = 6` informative columns concatenated with `K_noise = 18` autoregressive noise columns and shuffled. A successful search recovers the six signal columns from the 24 candidates.

## Prerequisites

- **Python ≥ 3.13.** A **free-threaded** build (3.13t or 3.14t) is strongly recommended — see the warning below.
- **A working idea of EDM.** If `lagged_embedding`, `(E, τ)`, and `simplex_projection` are new terms, skim [edmkit's *What is EDM?*](https://fujishigetemma.github.io/edmkit/concepts/edm/) and [*Embedding*](https://fujishigetemma.github.io/edmkit/concepts/embedding/) first.
- **NumPy fluency.** Most APIs take and return plain ndarrays.

## Installation

```bash
pip install edmkit-search
# or
uv add edmkit-search
```

edmkit-search depends on [`edmkit`](https://github.com/FujishigeTemma/edmkit) ≥ 0.0.9.

:::danger[Free-threaded Python required for parallel speedup]
The `ThreadPoolExecutor` pattern used throughout this tutorial gives a real CPU speedup **only** on a free-threaded ("no-GIL") Python build:

- **Python 3.13t** or **Python 3.14t** — the `t` suffix marks the free-threaded build.
- Launch with `PYTHON_GIL=0` to disable the GIL at runtime.

```bash
PYTHON_GIL=0 uv run python e2e/synthetic.py
```

On standard CPython the code still produces correct results but serializes on the GIL. A thirty-trajectory sweep that takes minutes on 3.14t can take an hour on 3.13. Install a free-threaded interpreter before any non-trivial sweep:

```bash
uv python install 3.14t
uv venv --python 3.14t
```
:::

## The mental model in one paragraph

A search is one loop over a **frontier** — a batch of column-subset candidates with their scores. Each iteration **expands** the frontier into children (a `Neighborhood`), **scores** the children (an `Energy`), and **keeps** the survivors (a `Strategy`). After the loop finishes, you read the trajectory and pick the step at which held-out predictive skill is highest.

A `State` is a 1D integer array — the columns of `X` currently selected. The search grows it by one index per step. **The search does not lag-embed the chosen columns**: each selected column contributes one dimension to the state vector at time `t`. Lagged embedding is a separate concern, used only in the pre-filter below.

## Step 1 — Frame predictive skill as energy

Strategies minimize energy. Wrap any "higher is better" goodness score as `1 - score`:

```python
import numpy as np
from edmkit.metrics import mean_rho


def corr(predictions: np.ndarray, observations: np.ndarray) -> np.ndarray:
    return 1.0 - mean_rho(predictions.reshape(observations.shape), observations)
```

`mean_rho` is mean Pearson correlation. `corr` is `1 − ρ̄`, which the strategy drives down. `mae` and `rmse` in `edmkit.metrics` are already losses.

## Step 2 — Wrap the data and carve the outer fold

```python
from edmkit.splits import temporal_fold

from edmkit.search.dataset import Dataset, Subset

data = Dataset(X=X_full, Y=Y_full)

outer = temporal_fold(data.X.shape[0], 0.8)
train = Subset(data, outer.train)
validation = Subset(data, outer.validation)
```

`Dataset` validates shape contracts and casts to `float32`. `Subset` is a non-copying view, and *is* a `Dataset`.

The **outer fold** is the held-out arm you evaluate the *whole search* on at the end. The energy is built from an **inner** split of `train` — never from `validation`. See [Validation](/edmkit-search/concepts/validation/) for why.

:::tip[Forecasting horizon]
To predict `Y` at `t + n` from `X` at `t`, shift the arrays before constructing the `Dataset`:

```python
n_ahead = 1
data = Dataset(X=X_full[:-n_ahead], Y=Y_full[n_ahead:])
```

See [Dataset](/edmkit-search/concepts/dataset/) for the convention.
:::

## Step 3 — Pre-filter weak candidates with `scan`

With `D = 24` candidates, many carry no information about `Y`. Drop them before the search: for each column, find the best lagged-embedding parameters `(E, τ)` and keep the column only if even its best embedding can predict `Y` above a threshold.

This is [`edmkit.embedding.scan`](https://fujishigetemma.github.io/edmkit/reference/embedding/) plus `select`:

```python
import os
from concurrent.futures import ThreadPoolExecutor
from edmkit.embedding import scan, select
from edmkit.simplex_projection import simplex_projection

E_range = list(range(1, 11))
tau_range = list(range(1, 6))
threshold = 0.1

with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
    def scan_one(i: int):
        return scan(
            train.X[:, i], train.Y,
            E=E_range, tau=tau_range,
            predict=simplex_projection, metric=mean_rho,
        )
    scanned = list(pool.map(scan_one, range(data.X.shape[1])))
    scores = np.array(
        [float(select(r, E=E_range, tau=tau_range)[2]) for r in scanned],
        dtype=np.float64,
    )
    mask = scores > threshold

data = Dataset(X=data.X[:, mask], Y=data.Y)
train = Subset(data, outer.train)
validation = Subset(data, outer.validation)
```

The pre-filter and the search **embed differently**:

- **Pre-filter (`scan`):** for each *single* column, build a lagged embedding `(E, τ)` and predict `Y` from it.
- **Search:** at each step, take the *current subset of columns* as the state vector at time `t` — no lag, one dimension per selected column.

That asymmetry matters when reusing parameters: `theiler_window` for [`energy.cross.loo`](/edmkit-search/concepts/energy/) defaults to `0` because the search does not lag-embed.

## Step 4 — Build the energy from an inner fold

The **energy** is what the search minimizes. The simplest scorer, `energy.cross.holdout`, fits on one inner-fold train arm and scores on that fold's validation arm:

```python
from edmkit.search import energy, neighborhood, state, strategy

inner = temporal_fold(train.X.shape[0], 0.75)
initial_context, plan = energy.cross.holdout(
    data=train,
    fold=inner,
    predict=simplex_projection,
    metric=corr,
    batch_size=64,
)
```

`energy.cross.holdout` does not return a callable directly. It returns `(initial_context, plan)`:

- `initial_context` is the energy's per-state state, threaded through the search. `holdout` carries width 0; `folds` carries the previous step's per-fold metric vector.
- `plan` is a factory: given a batch of states, it yields independent jobs.

Wrapping `plan` into a callable `Energy` is *your* choice — sequential, thread pool, process pool:

```python
def E(states, contexts):
    futures = [pool.submit(job) for job in plan(states, contexts)]
    n = states.shape[0]
    energies = np.empty(n, dtype=np.float64)
    new_contexts = np.empty((n, initial_context.shape[1]), dtype=np.float64)
    for f in futures:
        s, e, c = f.result()
        energies[s] = e
        new_contexts[s] = c
    return energies, new_contexts
```

This closure stays inside the `with ThreadPoolExecutor(...) as pool:` block from Step 3. See [Energy](/edmkit-search/concepts/energy/) for the same idiom with `loo` and `folds`.

## Step 5 — Compose neighborhood + strategy and run

```python
N = neighborhood.forward(data.X.shape[1])
S = strategy.greedy(E, N, depth=K_signal + 2)

initial = strategy.Frontier(
    states=state.initial(),                              # shape (1, 0): empty seed
    contexts=initial_context,                            # shape (1, K)
    energies=np.array([float("inf")], dtype=np.float64), # any real score beats infinity
)

trace = list(S(initial, np.random.default_rng(0)))
```

- `neighborhood.forward(n)` expands a state of length `d` into `n - d` children, each adding one index.
- `strategy.greedy(E, N, depth=D)` runs the whole search, committing to the single lowest-energy child at each depth, and yields one one-row frontier per depth — the best state found there.

After `K_signal + 2 = 8` depths, `trace[j].states[0]` is the selected subset of `j + 1` indices.

## Step 6 — Score the trajectory on the outer validation arm

The trace gives the search's *training-side* number. The honest *generalization-side* number comes from scoring the same indices against the **outer** validation arm:

```python
predictions = np.zeros((len(trace), *validation.Y.shape))
for j in range(len(trace)):
    selected = trace[j].states[0]
    predictions[j] = simplex_projection(
        train.X[:, selected], train.Y, validation.X[:, selected]
    ).reshape(validation.Y.shape)
val = corr(predictions, np.broadcast_to(validation.Y, predictions.shape))

best_step = int(np.argmin(val))
selected = trace[best_step].states[0]
```

Two streams come out of one search:

- `trace[j].energies[0]` — the **training** energy on the inner fold.
- `val[j]` — the **validation** score on the outer arm.

Training energy decreases monotonically by construction; validation flattens and eventually rises once the search starts overfitting. That curvature point is the natural stopping rule.

## Step 7 — Sweep configurations and compare

A single `(metric, energy, strategy)` choice is a starting point, not an answer. The point of three orthogonal callables is that you can swap one and compare:

```python
for metric_label, metric in [("Corr", corr), ("MAE", mae), ("RMSE", rmse)]:
    for energy_label, (initial_context, plan) in [
        ("holdout", energy.cross.holdout(data=train, fold=inner, predict=simplex_projection, metric=metric)),
        ("folds T=1", energy.cross.folds(data=train, folds=inner_folds, predict=simplex_projection, metric=metric, weight=energy.weight.softmax(1.0))),
        ("loo", energy.cross.loo(data=train, metric=metric)),
    ]:
        E = to_energy(initial_context, plan)
        for strategy_label, S in [
            ("greedy", strategy.greedy(E, N, depth=10)),
            ("beam", strategy.beam(E, N, width=1, depth=10, beams=3)),
        ]:
            trace = list(S(initial, np.random.default_rng(0)))
            # ... score trace against validation ...
```

Because the **outer fold is fixed across all runs**, the per-step validation curves are directly comparable. *Fix the data, vary one axis at a time, plot the curves on the same axes* — that is the canonical research workflow. See [Validation](/edmkit-search/concepts/validation/) for the full template.

## What's next

- [The Search Loop](/edmkit-search/concepts/search-loop/) — the loop, the frontier, and the three callables.
- [Energy](/edmkit-search/concepts/energy/) — the three built-in energies and the softmax-temperature knob.
- [Validation](/edmkit-search/concepts/validation/) — the two-level fold pattern and the sweep template.
- [`e2e/synthetic.py`](https://github.com/FujishigeTemma/edmkit-search/blob/main/e2e/synthetic.py) — the complete reference script.
