---
title: state
description: Search-state primitives.
sidebar:
  order: 2
---

**Type Aliases:**

Name | Description
---- | -----------
[`States`](#edmkit.search.state.states.States) | A batch of search states of shape ``(N, d)``: ``N`` states, each holding ``d`` indices into the original dataset. Within a single step, all states in the batch share the same length ``d``.

### `States` {#edmkit.search.state.states.States}

```python
type States = npt.NDArray[np.int64]
```

A batch of search states of shape ``(N, d)``: ``N`` states, each holding ``d`` indices into the original dataset. Within a single step, all states in the batch share the same length ``d``.

**Functions:**

Name | Description
---- | -----------
[`initial`](#initial) | Build the empty initial batch — a single zero-length state.

## `initial`

```python
initial() -> States
```

Build the empty initial batch — a single zero-length state.

**Returns:**

Type | Description
---- | -----------
<code>[States](#edmkit.search.state.states.States)</code> | Array of shape ``(1, 0)``.

