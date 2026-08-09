---
title: Neighborhood
description: What the expansion arm does, why it returns parents_idx, and how to write your own.
---

A `Neighborhood` is the expansion arm of the search loop. Given a batch of parent states, it produces the candidate children for the next step:

```python
type Neighborhood = Callable[[States, np.random.Generator], tuple[States, npt.NDArray[np.int64]]]
```

It returns two arrays:

- **`children`** of shape `(M, d')` — the next-step states (typically `d' = d + 1` for forward selection).
- **`parents_idx`** of shape `(M,)` — `parents_idx[i] in [0, N)` points to the parent row that produced `children[i]`.

## Why `parents_idx` exists

A neighborhood does not know about contexts — that is the energy's job. `parents_idx` is the back-pointer that lets a strategy thread parent-side data alongside the children without baking context-awareness into the neighborhood. See [Search Loop → The loop, in code](/edmkit-search/concepts/search-loop/#the-loop-in-code) for how `greedy` uses it.

## Forward selection

The one built-in is `forward(n)`, for variable selection from `D = n` candidate columns:

```python
from edmkit.search import neighborhood

N = neighborhood.forward(data.X.shape[1])
```

Each parent of length `d` (unique indices in `[0, n)`) expands into `n - d` children — one per index not yet selected. Forward selection therefore starts from the empty state `(1, 0)` and grows by one index per step.

Children of a single parent are emitted in a per-row *random* order from the supplied generator — combined with downstream stable sorts, this gives a per-parent random tie-break. One seed reproduces the whole sequence.

## Writing your own

A custom expander is any callable matching the contract. `States` is a plain `int64` ndarray, so most expanders are a few NumPy index manipulations — backward elimination, swap moves, or lattice neighborhoods over `(E, τ)` all fit.

