---
title: Energy
description: What an energy is, why it ships as a Plan, which built-in to use when, and how to write your own.
---

An `Energy` gives every candidate state in a batch a scalar score. Lower is better — wrap a goodness score as `corr = 1 - mean_rho` first.

```python
type Energy = Callable[[States, Contexts], tuple[Energies, Contexts]]
```

For each row of `states`, the energy returns one scalar in `energies` and one row of carry-forward state in `contexts`. The library minimizes `energies`; what `contexts` carries is up to the energy.

Examples on this page use:

```python
from edmkit.metrics import mean_rho

def corr(predictions, observations):
    return 1.0 - mean_rho(predictions.reshape(observations.shape), observations)
```

## The Plan / Energy split

Scoring is the most expensive operation in the search, and different environments parallelize differently (thread pool, process pool, serial). The built-in scorers therefore do **not** return an `Energy` directly. They return `(initial_context, plan)`:

```python
type Plan = Callable[[States, Contexts], Iterable[Callable[[], tuple[slice, Energies, Contexts]]]]
```

A `Plan` is a factory: given the current batch, it yields a sequence of independent zero-argument *jobs*. Each job returns `(slice, energies, contexts)` for the slice it owns. The library stays agnostic about *how* jobs execute.

## The canonical closure

The idiomatic `Energy` is a closure over `plan`, `initial_context`, and your executor:

```python
initial_context, plan = energy.cross.holdout(...)

with ThreadPoolExecutor() as pool:
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

    # ... strategy.run(...) inside the with-block
```

For sequential execution, replace the futures loop with `for job in plan(states, contexts): sl, e, c = job(); ...`. For a process pool, hand the jobs to `ProcessPoolExecutor.map`.

Requires free-threaded Python for parallel speedup — see [Getting Started → Installation](/edmkit-search/getting-started/#installation).

### Contexts: what threads between steps

`Contexts` is a `(N, K)` array threaded between steps. The width `K` is fixed at construction time and is opaque to the rest of the loop:

| Energy | `K` | What contexts carry |
| ------ | --- | ------------------- |
| `holdout` | 0 | (nothing) |
| `loo` | 0 | (nothing) |
| `folds` | `len(folds)` | the previous step's per-fold metric vector |

`folds` reports a weighted sum of `(metric - previous_metric)` across folds — see below.

### `batch_size`: what to tune

Each job materializes arrays of shape `(batch_size, T_inner, d)`. Cap `batch_size * T_inner * d` to fit in per-thread RAM. The default `10000` works for small inner folds; larger inner folds typically run with 2000–4000.

## Which energy when

| Energy | Best when | Cost vs `holdout` |
| ------ | --------- | ----------------- |
| **`holdout`** | The default. Inner-fold size is comfortable and one fold is representative. | 1× |
| **`folds`** | Inner-fold score is noisy across split positions, or you want to balance across regimes. | ~`len(folds)` × |
| **`loo`** | Training data is scarce; cannot afford a holdout fold. | depends on `T_inner` and `d` |

Start with `holdout`. Switch to `folds` if training-vs-validation trajectories disagree about the best stopping step. Switch to `loo` only when `len(train)` is small enough that a `0.75/0.25` inner split throws away too much.

## holdout — one fold, score directly

For each state, fit `predict` on the fold's train arm with the state's columns and score against the validation arm via `metric`. The metric value *is* the state's energy.

```python
from edmkit.simplex_projection import simplex_projection
from edmkit.splits import temporal_fold

inner = temporal_fold(train.X.shape[0], 0.75)
initial_context, plan = energy.cross.holdout(
    data=train, fold=inner,
    predict=simplex_projection, metric=corr,
    batch_size=64,
)
```

## loo — self-prediction with a Theiler window

Predict each row of the training arm from its in-library neighbours, excluding any within `theiler_window` time steps (blocking the trivial "next step is right next door" leak).

```python
initial_context, plan = energy.cross.loo(data=train, metric=corr, theiler_window=0)
```

:::note[`theiler_window` default is 0]
The conventional EDM recipe `(E - 1) * tau` applies only when columns are lag-embedded. This library's search does not lag-embed, so `theiler_window=0` is appropriate unless your `X` columns are themselves pre-embedded.
:::

## folds — multi-fold with per-fold attention

```python
from edmkit.splits import sliding_folds

T = train.X.shape[0]
inner_folds = sliding_folds(
    T,
    train_size=int(T * 0.4),
    validation_size=int(T * 0.2),
    stride=int(T * 0.2),
)
# [======t(0.4)======][=v(0.2)=]--------------------
# ----------[======t(0.4)======][=v(0.2)=]----------
# --------------------[======t(0.4)======][=v(0.2)=]

initial_context, plan = energy.cross.folds(
    data=train,
    folds=inner_folds,
    predict=simplex_projection,
    metric=corr,
    weight=energy.weight.softmax(temperature=1.0),
)
```

Each state is scored on every fold, producing a per-fold metric vector. The energy reported is:

```
energy = weight(previous_metrics) · (current_metrics − previous_metrics)
```

The `weight` function turns *previous* per-fold metrics into a row-stochastic weighting — a per-fold attention mechanism over the search trajectory. Current per-fold metrics are carried forward as the new context.

### softmax temperature as a knob

`energy.weight.softmax(temperature)` is a continuous slider:

- **Low temperature (`T → 0`)** concentrates weight on the folds with the largest previous metric (the parent's *worst* folds — "lower is better"). The search prioritises improving where the parent struggled most.
- **High temperature (`T → ∞`)** flattens toward uniform. The energy approaches the unweighted *mean* improvement across folds.

A sweep over `T ∈ {0.1, 1, 10}` is a natural ablation; see [`edmkit-search-experiments`](https://github.com/FujishigeTemma/edmkit-search-experiments).

## Writing your own energy

Return an `initial` context of shape `(1, K)` and a `plan(states, contexts)` yielding independent jobs (default-bind `start`/`end` in each job to avoid the closure trap). Copying [`energy/cross/holdout.py`](https://github.com/FujishigeTemma/edmkit-search/blob/main/src/edmkit/search/energy/cross/holdout.py) is the fastest path.
