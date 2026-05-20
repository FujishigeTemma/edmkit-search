---
title: Strategy
description: Per-step transitions, the immutable frontier, and the top-level runner.
---

A `Strategy` is what binds the other three pieces together into a per-iteration transition.

```python
type Step = Callable[
    [Frontier, np.random.Generator],
    Frontier,
]
```

Given the current frontier, a `Step` expands it via a `Neighborhood`, scores the children with an `Energy`, and selects which survive. The library ships two strategies and a runner.

## The Frontier

```python
@dataclass(frozen=True)
class Frontier:
    states: States      # (N, d)
    contexts: Contexts  # (N, K)
    energies: Energies  # (N,)
```

All three arrays are aligned along their leading axis: row `i` of `states` corresponds to `contexts[i]` and `energies[i]`. The frontier is frozen — every step produces a new one.

## beam

```python
step = strategy.beam(E, N, width=10, cutoff=float("inf"))
```

The general strategy: expand parents via `N`, score every child with `E`, drop children with energy strictly greater than `cutoff`, and keep the `width` survivors with the lowest energy.

The argsort under the hood is *stable*, so ties resolve in the order the neighborhood emitted the children. With [`forward`](/edmkit-search/reference/neighborhood/) (which randomizes within each parent), this means random tie-break — useful when the metric is coarse.

## greedy

```python
step = strategy.greedy(E, N, cutoff=float("inf"))
```

Sugar for `beam(E, N, width=1)`. Useful when you want a single trajectory rather than a frontier of candidates.

## run

```python
trace = list(
    strategy.run(
        initial_frontier,
        step,
        max_steps=8,
        rng=np.random.default_rng(0),
    )
)
```

`run` is the top-level driver. It iterates `step` from the initial frontier and **yields the single best survivor of each iteration** as a one-row frontier. Collecting the yields gives you the trajectory of best states across steps; the *full* post-step frontier is still threaded into the next iteration internally.

The iterator terminates early when `step` returns an empty frontier (the neighborhood has run out of children) or after `max_steps` iterations, whichever comes first.

## A Complete Step, Unpacked

For reference, this is what `beam.step` actually does:

```python
def step(frontier, rng):
    children, parents = N(frontier.states, rng)
    if children.shape[0] == 0:
        return Frontier(children, frontier.contexts[:0], np.empty(0))

    energies, contexts = E(children, frontier.contexts[parents])
    kept = np.flatnonzero(energies <= cutoff)
    order = kept[np.argsort(energies[kept], kind="stable")[:width]]

    return Frontier(
        states=children[order],
        contexts=contexts[order],
        energies=energies[order],
    )
```

The whole library's runtime cost lives inside `E(children, ...)`. Everything around it is index gymnastics.

## Initial Frontier

The conventional starting point for forward selection is the empty state, the energy's `initial_ctx`, and an infinite energy that any real score will beat:

```python
initial_ctx, plan = energy.holdout(...)
initial = strategy.Frontier(
    states=state.initial(),                                  # (1, 0)
    contexts=initial_ctx,                                    # (1, K)
    energies=np.array([float("inf")], dtype=np.float64),     # (1,)
)
```

For other neighborhoods (e.g. backward elimination from a full state) the starting frontier is whatever single state makes sense for that search. `run` does not constrain it.
