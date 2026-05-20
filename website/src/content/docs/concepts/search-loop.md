---
title: The Search Loop
description: The one loop and four abstractions that edmkit-search is built around.
---

edmkit-search is a small library wrapped around a single loop: build a trajectory of states by repeatedly expanding the current candidates, scoring the children, and keeping the best.

## The Loop

```
Frontier ──Neighborhood──▶  children  ──Energy──▶  (energies, contexts)
                                                          │
                                                          ▼
                                                       Strategy
                                                          │
                                                          ▼
                                                       Frontier'
```

Every step turns a `Frontier` into another `Frontier`. The four boxes — `Neighborhood`, `Energy`, `Strategy`, and the immutable `Frontier` itself — are the entirety of the library's vocabulary.

```python
@dataclass(frozen=True)
class Frontier:
    states: States      # (N, d)
    contexts: Contexts  # (N, K)
    energies: Energies  # (N,)
```

A `Step` is `(Frontier, rng) -> Frontier'`. The top-level [`run`](/edmkit-search/reference/strategy/) function iterates a step from an initial frontier and yields the best survivor of each iteration.

## What Each Piece Owns

| Piece | Type | Responsibility |
| ----- | ---- | -------------- |
| **State** | `(N, d)` ndarray | A batch of partial solutions. Each row is `d` indices into the dataset. |
| **Neighborhood** | `(parents, rng) -> (children, parents_idx)` | Expand each parent into its children, with parent-index back-pointers so callers can carry parent-side data forward. |
| **Energy** | `(states, contexts) -> (energies, contexts)` | Score the batch. Lower is better. The companion `Plan` form yields independent jobs for parallel execution. |
| **Strategy** | `Step` | Turn a frontier into the next one — expand, score, select. |

There is no base class. The four abstractions are plain callable protocols (and one `dataclass`), so you can drop in a custom implementation anywhere without subclassing.

## The Plan / Energy Split

The non-obvious design point is that **energies come in two forms**: a `Plan` (a factory of independent jobs) and a fully-applied `Energy` (a closure that scores a batch directly).

```python
type Plan = Callable[
    [States, Contexts],
    Iterable[Callable[[], tuple[slice, Energies, Contexts]]],
]

type Energy = Callable[[States, Contexts], tuple[Energies, Contexts]]
```

The three built-in scorers — [`holdout`](/edmkit-search/reference/energy/), [`loo`](/edmkit-search/reference/energy/), [`folds`](/edmkit-search/reference/energy/) — all return `(initial_context, plan)`. The caller decides how to execute the plan's jobs (sequentially, on a thread pool, on a process pool) and wraps the result back into an `Energy`. The search loop itself never touches a thread.

The idiomatic wrapper is a closure over `plan` and `pool`, defined alongside the search call:

```python
initial_context, plan = energy.holdout(...)

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

    # ... build N, S, initial, then strategy.run(initial, S, ...)
```

For a sequential run, replace the `pool.submit` line with a direct `for job in plan(...): s, e, c = job()` loop. See [`e2e/synthetic.py`](https://github.com/FujishigeTemma/edmkit-search/blob/main/e2e/synthetic.py) for the full pipeline.

## Energies Are Minimized

Strategies pick the rows with the *lowest* energy. When the natural quantity is a goodness score (e.g. Pearson correlation), wrap it as ``1 - score`` before handing it to an energy.

## Reproducibility

`run` accepts an `np.random.Generator` and threads it through every `Step` invocation. Neighborhoods may consume it for randomized expansion order (see [`forward`](/edmkit-search/reference/neighborhood/)); energies typically do not. A single seed reproduces the entire trajectory.
