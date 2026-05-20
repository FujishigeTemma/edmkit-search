---
title: strategy
description: Per-step transitions and the top-level search runner.
sidebar:
  order: 5
---

**Type Aliases:**

Name | Description
---- | -----------
[`Step`](#edmkit.search.strategy.frontier.Step) | A single search transition: ``(frontier, rng) -> frontier'``. The transition is responsible for expanding the frontier via a `Neighborhood`, scoring the children with an `Energy`, and selecting which survive.

### `Step` {#edmkit.search.strategy.frontier.Step}

```python
type Step = Callable[[Frontier, np.random.Generator], Frontier]
```

A single search transition: ``(frontier, rng) -> frontier'``. The transition is responsible for expanding the frontier via a `Neighborhood`, scoring the children with an `Energy`, and selecting which survive.

**Functions:**

Name | Description
---- | -----------
[`Frontier`](#Frontier) | Immutable batch of search candidates carried between steps.
[`beam`](#beam) | Build a beam-search `Step` that keeps the ``width`` lowest-energy children.
[`greedy`](#greedy) | Build a greedy `Step` that keeps only the single best child per parent.
[`run`](#run) | Iterate ``step`` from ``initial`` and yield the best survivor of each step.

## `Frontier`

```python
Frontier(states: States, contexts: Contexts, energies: Energies) -> None
```

Immutable batch of search candidates carried between steps.

The three arrays are aligned along their leading axis: row ``i`` of
``states`` corresponds to ``contexts[i]`` and ``energies[i]``. The
frontier is what each ``Step`` consumes and produces; ``run`` then
selects the single best row from each frontier to form the
trajectory.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`states`](#edmkit.search.strategy.frontier.Frontier.states) | <code>[States](#edmkit.search.state.States)</code> | Selected indices, shape ``(N, d)``.
[`contexts`](#edmkit.search.strategy.frontier.Frontier.contexts) | <code>[Contexts](#edmkit.search.energy.Contexts)</code> | Per-state energy context, shape ``(N, K)``.
[`energies`](#edmkit.search.strategy.frontier.Frontier.energies) | <code>[Energies](#edmkit.search.energy.Energies)</code> | Per-state energy values, shape ``(N,)``. Lower is better.



## `beam`

```python
beam(E: Energy, N: Neighborhood, *, width: int, cutoff: float = float('inf')) -> Step
```

Build a beam-search `Step` that keeps the ``width`` lowest-energy children.

On each invocation, the step expands the incoming frontier via
``N``, scores every child with ``E``, drops children whose energy
exceeds ``cutoff``, and then keeps the ``width`` survivors with
the lowest energy. The argsort is stable, so ties resolve in the
order ``N`` emitted the children — which, for the standard
`forward` neighborhood, means a per-row random tie-break.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`E` | <code>[Energy](#edmkit.search.energy.Energy)</code> | Energy used to score children. Must already be wired to its plan executor (see `Energy`). | *required*
`N` | <code>[Neighborhood](#edmkit.search.neighborhood.Neighborhood)</code> | Neighborhood used to expand parents. | *required*
`width` | <code>[int](#int)</code> | Number of children retained per step. Must be at least 1. ``width=1`` reduces this to `greedy`. | *required*
`cutoff` | <code>[float](#float)</code> | Children with energy strictly greater than ``cutoff`` are discarded before truncation. ``inf`` disables the cutoff. | <code>``float("inf")``</code>

**Returns:**

Type | Description
---- | -----------
<code>[Step](#edmkit.search.strategy.frontier.Step)</code> | ``(frontier, rng) -> frontier'`` returning a frontier of at most ``width`` rows. Returns an empty frontier when the neighborhood emits no children.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | If ``width < 1``.



## `greedy`

```python
greedy(E: Energy, N: Neighborhood, *, cutoff: float = float('inf')) -> Step
```

Build a greedy `Step` that keeps only the single best child per parent.

Equivalent to `beam` with ``width=1`` — the lowest-energy
child (subject to ``cutoff``) replaces the frontier on each step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`E` | <code>[Energy](#edmkit.search.energy.Energy)</code> | Energy used to score children. | *required*
`N` | <code>[Neighborhood](#edmkit.search.neighborhood.Neighborhood)</code> | Neighborhood used to expand parents. | *required*
`cutoff` | <code>[float](#float)</code> | Children with energy strictly greater than ``cutoff`` are discarded before selection. | <code>``float("inf")``</code>

**Returns:**

Type | Description
---- | -----------
<code>[Step](#edmkit.search.strategy.frontier.Step)</code> | ``(frontier, rng) -> frontier'`` returning a frontier of at most one row.



## `run`

```python
run(initial: Frontier, step: Step, *, max_steps: int, rng: np.random.Generator) -> Iterator[Frontier]
```

Iterate ``step`` from ``initial`` and yield the best survivor of each step.

On each iteration, ``step`` is applied to the current frontier;
the row with the lowest energy is yielded as a one-row frontier,
and the *full* post-step frontier feeds the next iteration. The
iterator terminates early when ``step`` returns an empty frontier
(i.e. the search has run out of candidates) or after ``max_steps``
iterations, whichever comes first.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`initial` | <code>[Frontier](#edmkit.search.strategy.frontier.Frontier)</code> | Starting frontier. For the standard forward-selection setup this is the empty state ``state.initial()`` paired with the energy's ``initial_context``. | *required*
`step` | <code>[Step](#edmkit.search.strategy.frontier.Step)</code> | Per-iteration transition (e.g. from `beam` or `greedy`). | *required*
`max_steps` | <code>[int](#int)</code> | Upper bound on the number of iterations. Must be non-negative. | *required*
`rng` | <code>[Generator](#numpy.random.Generator)</code> | Generator threaded into ``step`` (which in turn passes it to the neighborhood) so the whole search is reproducible from a single seed. | *required*

**Yields:**

Type | Description
---- | -----------
<code>[Frontier](#edmkit.search.strategy.frontier.Frontier)</code> | One-row frontier — the best survivor of each step. Collecting these yields the search trajectory.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | If ``max_steps`` is negative.

**Examples:**

```python
import numpy as np

from edmkit.search import energy, neighborhood, state, strategy

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

    N = neighborhood.forward(data.X.shape[1])
    S = strategy.greedy(E, N)
    initial = strategy.Frontier(
        states=state.initial(),
        contexts=initial_context,
        energies=np.array([float("inf")], dtype=np.float64),
    )

    trace = list(strategy.run(initial, S, max_steps=8, rng=np.random.default_rng(0)))
selected = trace[-1].states[0]
```

