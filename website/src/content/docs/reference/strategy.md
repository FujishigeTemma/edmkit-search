---
title: strategy
description: Per-step transitions and the top-level search runner.
sidebar:
  order: 5
---

**Type Aliases:**

Name | Description
---- | -----------
[`Strategy`](#edmkit.search.strategy.frontier.Strategy) | A full search: ``(initial, rng) -> trajectory``. Given the starting frontier, the strategy owns the whole loop — expanding states via a `Neighborhood`, scoring them with an `Energy`, and deciding which survive — and yields the best state found at each depth as a one-row frontier. Collecting the iterator yields the search trajectory.

### `Strategy` {#edmkit.search.strategy.frontier.Strategy}

```python
type Strategy = Callable[[Frontier, np.random.Generator], Iterator[Frontier]]
```

A full search: ``(initial, rng) -> trajectory``. Given the starting frontier, the strategy owns the whole loop — expanding states via a `Neighborhood`, scoring them with an `Energy`, and deciding which survive — and yields the best state found at each depth as a one-row frontier. Collecting the iterator yields the search trajectory.

**Functions:**

Name | Description
---- | -----------
[`Frontier`](#Frontier) | Immutable batch of search candidates.
[`beam`](#beam) | Build a chokudai-search `Strategy`.
[`greedy`](#greedy) | Build a greedy `Strategy` that commits to the single best child at each depth.

## `Frontier`

```python
Frontier(states: States, contexts: Contexts, energies: Energies) -> None
```

Immutable batch of search candidates.

The three arrays are aligned along their leading axis: row ``i`` of
``states`` corresponds to ``contexts[i]`` and ``energies[i]``. A
`Strategy` consumes an initial frontier and yields one-row
frontiers — the best state found at each depth — as the search
trajectory.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`states`](#edmkit.search.strategy.frontier.Frontier.states) | <code>[States](#edmkit.search.state.States)</code> | Selected indices, shape ``(N, d)``.
[`contexts`](#edmkit.search.strategy.frontier.Frontier.contexts) | <code>[Contexts](#edmkit.search.energy.Contexts)</code> | Per-state energy context, shape ``(N, K)``.
[`energies`](#edmkit.search.strategy.frontier.Frontier.energies) | <code>[Energies](#edmkit.search.energy.Energies)</code> | Per-state energy values, shape ``(N,)``. Lower is better.



## `beam`

```python
beam(E: Energy, N: Neighborhood, *, width: int, depth: int, beams: int, cutoff: float = float('inf')) -> Strategy
```

Build a chokudai-search `Strategy`.

Chokudai search keeps one candidate queue per depth, seeded with
the initial frontier at depth 0. Each *beam* is one pass over the
depths in order: it pops the ``width`` lowest-energy states from
the depth-``d`` queue, expands them via ``N``, scores the children
with ``E``, and pushes the survivors (those with energy at most
``cutoff``) into the depth-``d+1`` queue. Popped states never
return, so each additional beam expands the next-best states left
behind by earlier beams — a single beam is exactly classic beam
search of width ``width``, and every extra beam widens the search
around the depths where the earlier ones committed.

Within one depth, ties in the pop order resolve stably in
insertion order — which, for the standard `forward` neighborhood,
means a per-row random tie-break.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`E` | <code>[Energy](#edmkit.search.energy.Energy)</code> | Energy used to score children. Must already be wired to its plan executor (see `Energy`). | *required*
`N` | <code>[Neighborhood](#edmkit.search.neighborhood.Neighborhood)</code> | Neighborhood used to expand parents. | *required*
`width` | <code>[int](#int)</code> | Number of states popped per depth per beam. Must be at least 1. | *required*
`depth` | <code>[int](#int)</code> | Number of depths to search below the initial frontier. Must be non-negative. | *required*
`beams` | <code>[int](#int)</code> | Number of passes over the depth queues. Must be at least 1. ``beams=1`` reduces this to classic beam search of width ``width``; ``width=1, beams=1`` reduces it to `greedy`. | *required*
`cutoff` | <code>[float](#float)</code> | Children with energy strictly greater than ``cutoff`` are discarded and never enqueued. ``inf`` disables the cutoff. | <code>``float("inf")``</code>

**Returns:**

Type | Description
---- | -----------
<code>[Strategy](#edmkit.search.strategy.frontier.Strategy)</code> | ``(initial, rng) -> trajectory`` yielding a one-row frontier per depth ``1..depth`` — the lowest-energy state found at that depth across all beams. Iteration stops early at the first depth the search never reached (e.g. when the neighborhood emits no children). The ``rng`` is threaded into ``N`` so the whole search is reproducible from a single seed.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | If ``width < 1``, ``depth < 0``, or ``beams < 1``.



## `greedy`

```python
greedy(E: Energy, N: Neighborhood, *, depth: int, cutoff: float = float('inf')) -> Strategy
```

Build a greedy `Strategy` that commits to the single best child at each depth.

Equivalent to `beam` with ``width=1, beams=1`` — a single beam
that, at every depth, expands only the lowest-energy state
(subject to ``cutoff``) and never revisits the states it left
behind.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`E` | <code>[Energy](#edmkit.search.energy.Energy)</code> | Energy used to score children. | *required*
`N` | <code>[Neighborhood](#edmkit.search.neighborhood.Neighborhood)</code> | Neighborhood used to expand parents. | *required*
`depth` | <code>[int](#int)</code> | Number of depths to search below the initial frontier. Must be non-negative. | *required*
`cutoff` | <code>[float](#float)</code> | Children with energy strictly greater than ``cutoff`` are discarded before selection. ``inf`` disables the cutoff. | <code>``float("inf")``</code>

**Returns:**

Type | Description
---- | -----------
<code>[Strategy](#edmkit.search.strategy.frontier.Strategy)</code> | ``(initial, rng) -> trajectory`` yielding a one-row frontier per depth ``1..depth``.

