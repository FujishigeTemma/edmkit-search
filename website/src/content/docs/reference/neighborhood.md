---
title: neighborhood
description: Expanders that map a batch of parent states to children.
sidebar:
  order: 3
---

**Type Aliases:**

Name | Description
---- | -----------
[`Neighborhood`](#edmkit.search.neighborhood.neighborhood.Neighborhood) | A function ``(parents, rng) -> (children, parents_idx)`` that expands a batch of ``N`` parent states into ``M`` children.

### `Neighborhood` {#edmkit.search.neighborhood.neighborhood.Neighborhood}

```python
type Neighborhood = Callable[[States, np.random.Generator], tuple[States, npt.NDArray[np.int64]]]
```

A function ``(parents, rng) -> (children, parents_idx)`` that expands a batch of ``N`` parent states into ``M`` children.

Given ``parents`` of shape ``(N, d)``, returns:

* ``children`` of shape ``(M, d')`` — the next-step states (typically ``d' = d + 1``).
* ``parents_idx`` of shape ``(M,)`` — ``parents_idx[i] in [0, N)`` points to the parent row
  that produced ``children[i]``.

The ``parents_idx`` array lets callers (e.g. `beam`) replicate parent-side data such as
the energy context in lockstep with the children, without the neighborhood having to know
about that data.

**Functions:**

Name | Description
---- | -----------
[`forward`](#forward) | Build a forward-selection `Neighborhood` over ``n`` candidate indices.

## `forward`

```python
forward(n: int) -> Neighborhood
```

Build a forward-selection `Neighborhood` over ``n`` candidate indices.

Each parent state of length ``d`` (assumed to hold unique indices
in ``[0, n)``) expands into exactly ``n - d`` children — one per
index not yet selected. Within a single parent, the order of the
emitted children is randomized via the supplied generator so that
downstream truncations (e.g. beam ``width``) do not systematically
favour low indices; across parents, the parent order is preserved.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n` | <code>[int](#int)</code> | Size of the candidate universe. Must be non-negative. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[Neighborhood](#edmkit.search.neighborhood.neighborhood.Neighborhood)</code> | ``(parents, rng) -> (children, parents_idx)`` with ``children.shape == (N * (n - d), d + 1)``.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | If ``n`` is negative.

