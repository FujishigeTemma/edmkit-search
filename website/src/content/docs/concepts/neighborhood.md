---
title: Neighborhood
description: Expanding a batch of parent states into a batch of children.
---

A `Neighborhood` is the expansion arm of the search loop. Given a batch of parent states, it produces the candidate children for the next step.

```python
type Neighborhood = Callable[
    [States, np.random.Generator],
    tuple[States, npt.NDArray[np.int64]],
]
```

A neighborhood returns two arrays:

- **`children`** of shape `(M, d')` — the next-step states (typically `d' = d + 1`).
- **`parents_idx`** of shape `(M,)` — `parents_idx[i] in [0, N)` points to the parent row that produced `children[i]`.

## Why parents_idx Matters

A neighborhood does not know about *contexts* — that is the energy's job. But strategies need to replicate parent-side data alongside the children when expanding the frontier. `parents_idx` is the back-pointer that makes this possible without baking context-awareness into the neighborhood:

```python
# Inside beam.step:
children, parents = N(frontier.states, rng)
energies, contexts = E(children, frontier.contexts[parents])
#                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                       parents-side data lifted onto children
```

This is the canonical pattern: any per-parent data — contexts, accumulated metrics, attribution — can ride along by indexing with `parents`. The neighborhood stays agnostic.

## Forward Selection

The one built-in neighborhood is [`forward(n)`](/edmkit-search/reference/neighborhood/):

```python
from edmkit.search import neighborhood

N = neighborhood.forward(data.X.shape[1])
```

Each parent state of length `d` (assumed to hold unique indices in `[0, n)`) expands into exactly `n - d` children — one per index not yet selected. So a forward-selection search starts from the empty state `(1, 0)` and grows the state by one index per step.

### Random Tie-Break Within Parents

The children of a single parent are emitted in a per-row *random* order, derived from the supplied `np.random.Generator`. Across parents, the parent order is preserved.

This matters because downstream truncations are stable sorts. When a beam strategy keeps the `width` lowest-energy children and several children tie, the random emission order inside a parent prevents systematic favoritism toward low indices. With a seeded generator the whole sequence is still reproducible.

## Writing Your Own Neighborhood

A custom expander is just a callable that respects the protocol. Two examples of how you might extend the library:

- **Backward elimination.** Start from the full state `np.arange(n)[None, :]` and emit `d` children per parent, each one omitting one index. `parents_idx` is `np.repeat(np.arange(N), d)`.
- **Swap moves.** For each parent of length `d`, emit `d * (n - d)` children — one per (in-state, out-of-state) pair. Useful for local search around a fixed state size.

The library does not prescribe what "expansion" means; anything that respects the `(children, parents_idx)` contract works.
