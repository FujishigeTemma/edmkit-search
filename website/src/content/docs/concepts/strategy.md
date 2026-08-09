---
title: Strategy
description: What a strategy is, why greedy and beam are different operating points, and how a strategy owns the search loop.
---

A `Strategy` binds a `Neighborhood` and an `Energy` into a full search:

```python
type Strategy = Callable[[Frontier, np.random.Generator], Iterator[Frontier]]
```

Given the initial frontier, a strategy owns the whole loop — expanding states via the `Neighborhood`, scoring them with the `Energy`, deciding which survive — and yields the best state found at each depth as a one-row frontier. Collecting the iterator yields the search trajectory. The library ships two factories, `greedy` and `beam`.

## Why two strategies

- **`greedy(depth=D)`** commits to the single best child at every depth. Cheapest possible, but a wrong early commit cannot be undone.
- **`beam(width=W, depth=D, beams=B)`** runs **chokudai search**. A single beam is exactly classic beam search of width `W`; every extra beam goes back and expands the next-best states the earlier beams left behind, so the search widens precisely at the depths where the earlier beams committed. Costs ~`W×B` more than greedy, but can recover from an early choice that turns out subdominant.

`greedy` is sugar for `beam(width=1, beams=1)` — one beam of width 1 *is* greedy selection.

The canonical comparison is `{greedy, beam(width=1, beams=3)}` on the same `(Energy, Neighborhood)`. If the beam's per-step validation curve is consistently below greedy's, the energy landscape is rugged enough to justify the extra cost.

## beam — chokudai search

```python
S = strategy.beam(E, N, width=1, depth=8, beams=3, cutoff=float("inf"))
trace = list(S(initial_frontier, np.random.default_rng(0)))
```

`beam` keeps one candidate queue per depth, seeded with the initial frontier at depth 0. Each *beam* is one pass over the depths in order:

```python
for _ in range(beams):
    for d in range(depth):
        parents = pop_lowest(queues[d], width)                    # popped states never return
        children, parents_idx = N(parents.states, rng)            # (M, d+1), (M,)
        energies, contexts = E(children, parents.contexts[parents_idx])  # (M,), (M, K)
        push(queues[d + 1], survivors_below(cutoff))              # feed the next depth
```

The whole library's runtime cost lives inside `E(children, ...)`. Everything around it is index gymnastics on the frozen `Frontier` (defined in [Search Loop](/edmkit-search/concepts/search-loop/)).

Popped states never return, so:

- **`beams=1`** reduces to classic beam search of width `width` — each depth expands its top-`width` states exactly once.
- **`beams=B`** re-visits every depth `B` times, expanding the next-best leftovers. The extra budget concentrates where the earlier beams' commitments were tightest, which is the property that makes chokudai search a strong anytime refinement of a fixed-width beam.

Within one depth, pop-order ties resolve stably in insertion order — combined with [`forward`](/edmkit-search/concepts/neighborhood/)'s per-parent shuffle this yields a random tie-break.

The trajectory contains, for each depth `1..depth`, the lowest-energy state found at that depth across all beams. Iteration stops early at the first depth the search never reached (e.g. when the neighborhood emits no children), so **`trace[j].states[0]` is the selected index set of length `j + 1`** regardless of `width` and `beams`.

## greedy

```python
S = strategy.greedy(E, N, depth=8, cutoff=float("inf"))
trace = list(S(initial_frontier, np.random.default_rng(0)))
```

Sugar for `beam(E, N, width=1, depth=8, beams=1)`. Useful when you want a single cheap trajectory.

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

The strategy does not constrain the initial frontier — backward elimination from a full state works the same way.

## Writing your own

Any callable matching `Strategy = (initial, rng) -> Iterator[Frontier]` works (tournament selection, mutation, restart-on-stall, etc.). Yield one-row frontiers so downstream trajectory-scoring code stays uniform, respect the `rng`, do not capture mutable state across calls.
