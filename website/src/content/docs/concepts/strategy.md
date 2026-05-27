---
title: Strategy
description: What a strategy is, why greedy and beam are different operating points, and how the runner drives them.
---

A `Strategy` binds a `Neighborhood` and an `Energy` into one per-iteration transition.

```python
type Step = Callable[[Frontier, np.random.Generator], Frontier]
```

Given the current frontier, a `Step` expands it via the `Neighborhood`, scores the children with the `Energy`, and selects which survive. The library ships `greedy`, `beam`, and a runner `run`.

## Why two strategies

- **`greedy`** commits to the single best child at every step. Cheapest possible, but a wrong early commit cannot be undone.
- **`beam(width=W)`** keeps the `W` best children. Costs ~`W×` more, but can recover from an early choice that turns out subdominant.

The canonical comparison is `{greedy, beam(width=3)}` on the same `(Energy, Neighborhood)`. If the beam's per-step validation curve is consistently below greedy's, the energy landscape is rugged enough to justify the extra cost.

## The per-step transition

The `Frontier` (defined in [Search Loop](/edmkit-search/concepts/search-loop/)) is the loop's single piece of state. This is what `beam`'s step actually does:

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

The whole library's runtime cost lives inside `E(children, ...)`.

### beam

```python
step = strategy.beam(E, N, width=10, cutoff=float("inf"))
```

Expand via `N`, score every child with `E`, drop children whose energy is strictly greater than `cutoff`, and keep the `width` survivors with the lowest energy. The argsort is *stable*, so ties resolve in the order the neighborhood emitted them — combined with [`forward`](/edmkit-search/concepts/neighborhood/) this yields a per-parent random tie-break.

### greedy

```python
step = strategy.greedy(E, N, cutoff=float("inf"))
```

Sugar for `beam(E, N, width=1)`. Useful when you want a single trajectory.

## run — the top-level driver

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

`run` iterates `step` from the initial frontier and **yields the single best survivor of each iteration** as a one-row frontier. The *full* post-step frontier is threaded into the next iteration internally.

The iterator terminates when `step` returns an empty frontier or after `max_steps` iterations.

If you need the whole beam at each step, call `step` directly in your own loop.

## Initial frontier

For forward selection, the starting frontier is the empty state, the energy's `initial_context`, and an infinite energy that any real score beats:

```python
initial_context, plan = energy.cross.holdout(...)
initial = strategy.Frontier(
    states=state.initial(),                                  # (1, 0)
    contexts=initial_context,                                # (1, K)
    energies=np.array([float("inf")], dtype=np.float64),     # (1,)
)
```

`run` does not constrain the initial frontier — backward elimination from a full state works the same way.

## Writing your own

Any callable matching `Step = (Frontier, rng) -> Frontier'` works (tournament selection, mutation, restart-on-stall, etc.). Return a `Frontier`, respect the `rng`, do not capture mutable state.

