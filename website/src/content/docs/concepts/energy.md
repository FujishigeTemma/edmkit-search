---
title: Energy
description: Scoring batches of states, the Plan/Energy split, and the three built-in energies.
---

An `Energy` is a function `(states, contexts) -> (energies, contexts')` that gives every state in a batch a scalar score. Lower is better.

## Two Forms: Plan and Energy

The three built-in scorers — [`holdout`](/edmkit-search/reference/energy/), [`loo`](/edmkit-search/reference/energy/), [`folds`](/edmkit-search/reference/energy/) — do **not** return an `Energy` directly. They return `(initial_context, plan)`:

```python
type Plan = Callable[
    [States, Contexts],
    Iterable[Callable[[], tuple[slice, Energies, Contexts]]],
]
```

A `Plan` is a factory that, given the current batch, yields a sequence of *jobs*. Each job is a zero-argument callable that returns `(slice, energies, contexts)` for the contiguous slice of the batch it owns.

The caller picks an execution strategy and folds the per-job results back into an `Energy`:

```python
def parallel(initial_ctx, plan, pool):
    def E(states, contexts):
        futures = [pool.submit(job) for job in plan(states, contexts)]
        n = states.shape[0]
        energies = np.empty(n)
        new_ctx  = np.empty((n, initial_ctx.shape[1]))
        for f in futures:
            s, e, c = f.result()
            energies[s] = e
            new_ctx[s]  = c
        return energies, new_ctx
    return E
```

The library never touches a thread. That decision is yours.

## Contexts

`Contexts` is a `(N, K)` array threaded between steps alongside `states` and `energies`. The width `K` is fixed by the energy at construction time and is opaque to the rest of the search loop. Built-in scorers use it as follows:

| Energy | `K` | What contexts carry |
| ------ | --- | ------------------- |
| `holdout` | 0 | (nothing) |
| `loo` | 0 | (nothing) |
| `folds` | `len(folds)` | the previous step's per-fold metric vector |

`folds` is the only built-in that uses contexts non-trivially. The energy it reports is the weighted sum of `(metric - previous_metric)` across folds, so the search is driven by the *delta* relative to the parent state — see below.

## holdout: One Fold, Score Directly

```python
initial_ctx, plan = energy.holdout(
    data=train,
    fold=fold,
    predict=simplex_projection,
    metric=mean_rho,           # lower is better
    batch_size=64,
)
```

For each state in the batch, the columns of `train.X` indexed by the state are used to fit `predict` on the fold's train arm and to score it against the validation arm via `metric`. The metric value *is* the state's energy. The simplest and fastest of the three.

## loo: Self-Prediction with a Theiler Window

```python
initial_ctx, plan = energy.loo(
    data=train,
    metric=mean_rho,
    theiler_window=(E - 1) * tau,
)
```

For each state, run leave-one-out simplex projection on `train.X[:, state]` against `train.Y`. Each library point is predicted from its in-library neighbours, excluding temporally close points via the Theiler window (`(E - 1) * tau` is the conventional choice for lagged embeddings). Useful when you do not want to commit a holdout fold.

## folds: Multi-Fold with Attention

```python
initial_ctx, plan = energy.folds(
    data=train,
    folds=cv_folds,                 # Sequence[Fold]
    predict=simplex_projection,
    metric=mean_rho,
    weight=softmax(temperature=1.0),
)
```

Each state is scored on every fold, producing a per-fold metric vector. The energy reported is:

```
weight(previous_metrics) · (current_metrics − previous_metrics)
```

The `weight` function (e.g. [`softmax`](/edmkit-search/reference/energy/)) turns the *previous* per-fold metrics into a row-stochastic weighting — a per-fold attention mechanism over the search trajectory. The current per-fold metrics are then carried forward as the new context.

Use this when you want the search to focus on folds where the parent state is already doing well (low temperature on a "lower is better" metric → concentrate on the folds with the smallest gap), or to penalize regression on previously-strong folds.

## Writing Your Own Energy

A custom energy needs to do two things:

1. Return an `initial` context of shape `(1, K)` for the chosen `K`.
2. Provide a `plan(states, contexts)` that yields independent jobs.

Each job is a closure that captures its slice — by default-binding `start` and `end` you avoid the Python loop-variable closure trap. The slice it owns is `slice(start, end)`, and it must return energies and contexts of that size.

The three built-ins all follow this template; copying the structure of [`energy/holdout.py`](https://github.com/FujishigeTemma/edmkit-search/blob/main/src/edmkit/search/energy/holdout.py) is the fastest way to a custom scorer.
