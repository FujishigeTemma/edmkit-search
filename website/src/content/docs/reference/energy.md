---
title: energy
description: Scoring functions for batches of states, with deferred-job plans.
sidebar:
  order: 4
---

**Type Aliases:**

Name | Description
---- | -----------
[`Contexts`](#edmkit.search.energy.energy.Contexts) | 2D array of shape ``(N, K)`` carrying per-state auxiliary information between steps. The width ``K`` is fixed by the energy at construction time (``0`` when no context is needed); the contents are opaque to the rest of the search loop.
[`Energies`](#edmkit.search.energy.energy.Energies) | 1D array of energy values of shape ``(N,)``, one per state. Lower is better — strategies minimize energy.
[`Energy`](#edmkit.search.energy.energy.Energy) | A fully-applied energy: given a batch of states and their incoming contexts, return the new energies and contexts. Construct one by executing a `Plan` (e.g. via a thread pool).
[`Plan`](#edmkit.search.energy.energy.Plan) | A factory that, given ``(states, contexts)``, yields a sequence of jobs covering the batch.
[`WeightFunc`](#edmkit.search.energy.weight.WeightFunc) | A weighting function ``(N, K) -> (N, K)`` that turns a context matrix into per-row weights. Rows are expected to sum to 1 so the result behaves as an attention distribution over the K columns.

### `Contexts` {#edmkit.search.energy.energy.Contexts}

```python
type Contexts = npt.NDArray[np.float64]
```

2D array of shape ``(N, K)`` carrying per-state auxiliary information between steps. The width ``K`` is fixed by the energy at construction time (``0`` when no context is needed); the contents are opaque to the rest of the search loop.

### `Energies` {#edmkit.search.energy.energy.Energies}

```python
type Energies = npt.NDArray[np.float64]
```

1D array of energy values of shape ``(N,)``, one per state. Lower is better — strategies minimize energy.

### `Energy` {#edmkit.search.energy.energy.Energy}

```python
type Energy = Callable[[States, Contexts], tuple[Energies, Contexts]]
```

A fully-applied energy: given a batch of states and their incoming contexts, return the new energies and contexts. Construct one by executing a `Plan` (e.g. via a thread pool).

### `Plan` {#edmkit.search.energy.energy.Plan}

```python
type Plan = Callable[[States, Contexts], Iterable[Callable[[], tuple[slice, Energies, Contexts]]]]
```

A factory that, given ``(states, contexts)``, yields a sequence of jobs covering the batch.

Each job is a zero-argument callable that returns ``(slice, energies, contexts)`` for the
contiguous slice it owns. Splitting work into independent jobs lets the caller execute them
however they like — sequentially, on a thread pool, on a process pool — without the plan
itself needing to know.

### `WeightFunc` {#edmkit.search.energy.weight.WeightFunc}

```python
type WeightFunc = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
```

A weighting function ``(N, K) -> (N, K)`` that turns a context matrix into per-row weights. Rows are expected to sum to 1 so the result behaves as an attention distribution over the K columns.

**Functions:**

Name | Description
---- | -----------
[`cross`](#cross) | 
[`softmax`](#softmax) | Build a row-wise softmax `WeightFunc` with the given temperature.
[`within`](#within) | 

## `cross`

**Modules:**

Name | Description
---- | -----------
[`folds`](#edmkit.search.energy.cross.folds) | 
[`holdout`](#edmkit.search.energy.cross.holdout) | 
[`loo`](#edmkit.search.energy.cross.loo) | 

### `folds`

**Functions:**

Name | Description
---- | -----------
[`folds`](#edmkit.search.energy.cross.folds.folds) | Build a cross-target multi-fold `Plan` that scores weighted stepwise improvement.

#### `folds`

```python
folds(*, data: dataset.Dataset, folds: Sequence[Fold], predict: PredictFunc, metric: MetricFunc, weight: WeightFunc, batch_size: int = 10000) -> tuple[Contexts, Plan]
```

Build a cross-target multi-fold `Plan` that scores weighted stepwise improvement.

Each state predicts the fixed target ``data.Y`` on every fold to
produce a per-fold metric vector. The energy reported for the
state is the ``weight``-ed sum of ``(metric - previous_metric)``
across folds, so the search is driven by the delta relative to
the parent state. The per-fold metric vector is then carried
forward as the new context.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Dataset](#edmkit.search.dataset.Dataset)</code> | Dataset whose columns are selected by each state. | *required*
`folds` | <code>[Sequence](#collections.abc.Sequence)[[Fold](#edmkit.splits.Fold)]</code> | Folds to score on. Must be non-empty. | *required*
`predict` | <code>[PredictFunc](#edmkit.types.PredictFunc)</code> | Prediction function with signature ``(X, Y, Q) -> predictions``. | *required*
`metric` | <code>[MetricFunc](#edmkit.metrics.MetricFunc)</code> | Per-fold reducer. Lower must mean better. | *required*
`weight` | <code>[WeightFunc](#edmkit.search.energy.weight.WeightFunc)</code> | Function ``(N, K) -> (N, K)`` producing per-fold weights from the incoming per-fold context. | *required*
`batch_size` | <code>[int](#int)</code> | Number of states processed in a single job. | <code>10000</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`initial` | <code>[Contexts](#edmkit.search.energy.energy.Contexts)</code> | Initial context of shape ``(1, K)`` filled with zeros, where ``K = len(folds)``.
`plan` | <code>[Plan](#edmkit.search.energy.energy.Plan)</code> | Plan that yields one job per ``batch_size`` chunk of states.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | If ``folds`` is empty.

### `holdout`

**Functions:**

Name | Description
---- | -----------
[`holdout`](#edmkit.search.energy.cross.holdout.holdout) | Build a cross-target holdout `Plan` that scores each state on a single fold.

#### `holdout`

```python
holdout(*, data: dataset.Dataset, fold: Fold, predict: PredictFunc, metric: MetricFunc, batch_size: int = 10000) -> tuple[Contexts, Plan]
```

Build a cross-target holdout `Plan` that scores each state on a single fold.

For each state in the batch, the columns of ``data.X`` indexed by
the state are used to fit ``predict`` on the fold's train arm and
score it against the fixed target ``data.Y`` on the fold's
validation arm via ``metric``. The metric value becomes the
state's energy directly; no carry-over context is needed.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Dataset](#edmkit.search.dataset.Dataset)</code> | Dataset whose columns (``X[:, state]``) are selected by each state. | *required*
`fold` | <code>[Fold](#edmkit.splits.Fold)</code> | Train/validation split used to score every state. | *required*
`predict` | <code>[PredictFunc](#edmkit.types.PredictFunc)</code> | Prediction function with signature ``(X, Y, Q) -> predictions``. | *required*
`metric` | <code>[MetricFunc](#edmkit.metrics.MetricFunc)</code> | Reducer turning predictions and observations into a scalar per state. Lower must mean better. | *required*
`batch_size` | <code>[int](#int)</code> | Number of states processed in a single job. | <code>10000</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`initial` | <code>[Contexts](#edmkit.search.energy.energy.Contexts)</code> | Initial context of shape ``(1, 0)``.
`plan` | <code>[Plan](#edmkit.search.energy.energy.Plan)</code> | Plan that yields one job per ``batch_size`` chunk of states.

### `loo`

**Functions:**

Name | Description
---- | -----------
[`loo`](#edmkit.search.energy.cross.loo.loo) | Build a cross-target leave-one-out `Plan` that scores each state.

#### `loo`

```python
loo(*, data: dataset.Dataset, metric: MetricFunc, theiler_window: int = 0, batch_size: int = 10000) -> tuple[Contexts, Plan]
```

Build a cross-target leave-one-out `Plan` that scores each state.

For each state in the batch, the corresponding column-subset of
``data.X`` is used as the library for simplex-projection LOO; each
library point is predicted from its in-library neighbours
(excluding temporally close points via the Theiler window), and
the predictions are scored against ``data.Y`` with ``metric``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Dataset](#edmkit.search.dataset.Dataset)</code> | Dataset whose columns are selected by each state. | *required*
`metric` | <code>[MetricFunc](#edmkit.metrics.MetricFunc)</code> | Reducer turning predictions and observations into a scalar per state. Lower must mean better. | *required*
`theiler_window` | <code>[int](#int)</code> | Theiler window half-width passed to ``simplex_projection.loo``. | <code>0</code>
`batch_size` | <code>[int](#int)</code> | Number of states processed in a single job. | <code>10000</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`initial` | <code>[Contexts](#edmkit.search.energy.energy.Contexts)</code> | Initial context of shape ``(1, 0)``.
`plan` | <code>[Plan](#edmkit.search.energy.energy.Plan)</code> | Plan that yields one job per ``batch_size`` chunk of states.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | If ``theiler_window`` is negative.



## `softmax`

```python
softmax(temperature: float = 1.0) -> WeightFunc
```

Build a row-wise softmax `WeightFunc` with the given temperature.

Lower temperature concentrates weight on the columns with the
largest values; higher temperature flattens toward a uniform
distribution. The implementation subtracts the per-row maximum
before exponentiating, so it is numerically stable for any input
scale.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`temperature` | <code>[float](#float)</code> | Positive scalar controlling the sharpness of the distribution. Must be strictly positive. | <code>1.0</code>

**Returns:**

Type | Description
---- | -----------
<code>[WeightFunc](#edmkit.search.energy.weight.WeightFunc)</code> | ``(N, K) -> (N, K)`` row-stochastic weighting.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | If ``temperature`` is not positive.



## `within`

**Modules:**

Name | Description
---- | -----------
[`folds`](#edmkit.search.energy.within.folds) | 
[`holdout`](#edmkit.search.energy.within.holdout) | 
[`loo`](#edmkit.search.energy.within.loo) | 

### `folds`

**Functions:**

Name | Description
---- | -----------
[`folds`](#edmkit.search.energy.within.folds.folds) | Build a multi-fold `Plan` for within-state prediction.

#### `folds`

```python
folds(*, data: dataset.Dataset, folds: Sequence[Fold], predict: PredictFunc, metric: MetricFunc, weight: WeightFunc, batch_size: int = 10000) -> tuple[Contexts, Plan]
```

Build a multi-fold `Plan` for within-state prediction.

This is the within-state analogue of ``energy.cross.folds``: each
state is scored on every fold, and the reported energy is the
weighted delta from the incoming per-fold context. Each state
forecasts its own selected columns: ``X[:, state] -> Y[:, state]``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Dataset](#edmkit.search.dataset.Dataset)</code> | Dataset whose ``X`` columns are candidate state coordinates. | *required*
`folds` | <code>[Sequence](#collections.abc.Sequence)[[Fold](#edmkit.splits.Fold)]</code> | Folds to score on. Must be non-empty. | *required*
`predict` | <code>[PredictFunc](#edmkit.types.PredictFunc)</code> | Prediction function with signature ``(X, Y, Q) -> predictions``. | *required*
`metric` | <code>[MetricFunc](#edmkit.metrics.MetricFunc)</code> | Per-fold reducer. Lower must mean better. | *required*
`weight` | <code>[WeightFunc](#edmkit.search.energy.weight.WeightFunc)</code> | Function ``(N, K) -> (N, K)`` producing per-fold weights from incoming contexts. | *required*
`batch_size` | <code>[int](#int)</code> | Number of states processed in a single job. | <code>10000</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`initial` | <code>[Contexts](#edmkit.search.energy.energy.Contexts)</code> | Initial context of shape ``(1, K)`` filled with zeros, where ``K = len(folds)``.
`plan` | <code>[Plan](#edmkit.search.energy.energy.Plan)</code> | Plan that yields one job per ``batch_size`` chunk of states.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | If ``folds`` is empty.

### `holdout`

**Functions:**

Name | Description
---- | -----------
[`holdout`](#edmkit.search.energy.within.holdout.holdout) | Build a holdout-validation `Plan` for within-state prediction.

#### `holdout`

```python
holdout(*, data: dataset.Dataset, fold: Fold, predict: PredictFunc, metric: MetricFunc, batch_size: int = 10000) -> tuple[Contexts, Plan]
```

Build a holdout-validation `Plan` for within-state prediction.

For each state, the selected columns of ``data.X`` are the library/
query coordinates and the *same* selected columns of ``data.Y`` are
the prediction target: ``X[:, state] -> Y[:, state]``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Dataset](#edmkit.search.dataset.Dataset)</code> | Dataset whose ``X`` columns are candidate state coordinates. | *required*
`fold` | <code>[Fold](#edmkit.splits.Fold)</code> | Train/validation split used to score every state. | *required*
`predict` | <code>[PredictFunc](#edmkit.types.PredictFunc)</code> | Prediction function with signature ``(X, Y, Q) -> predictions``. | *required*
`metric` | <code>[MetricFunc](#edmkit.metrics.MetricFunc)</code> | Reducer turning predictions and observations into a scalar per state. Lower must mean better. | *required*
`batch_size` | <code>[int](#int)</code> | Number of states processed in a single job. | <code>10000</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`initial` | <code>[Contexts](#edmkit.search.energy.energy.Contexts)</code> | Initial context of shape ``(1, 0)``.
`plan` | <code>[Plan](#edmkit.search.energy.energy.Plan)</code> | Plan that yields one job per ``batch_size`` chunk of states.

### `loo`

**Functions:**

Name | Description
---- | -----------
[`loo`](#edmkit.search.energy.within.loo.loo) | Build a leave-one-out `Plan` for within-state prediction.

#### `loo`

```python
loo(*, data: dataset.Dataset, metric: MetricFunc, theiler_window: int = 0, batch_size: int = 10000) -> tuple[Contexts, Plan]
```

Build a leave-one-out `Plan` for within-state prediction.

For each state, the selected columns of ``data.X`` are predicted
from their in-library neighbours and scored against the *same*
selected columns of ``data.Y`` (``X[:, state] -> Y[:, state]``).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Dataset](#edmkit.search.dataset.Dataset)</code> | Dataset whose ``X`` columns are candidate state coordinates. | *required*
`metric` | <code>[MetricFunc](#edmkit.metrics.MetricFunc)</code> | Reducer turning predictions and observations into a scalar per state. Lower must mean better. | *required*
`theiler_window` | <code>[int](#int)</code> | Theiler window half-width passed to ``simplex_projection.loo``. | <code>0</code>
`batch_size` | <code>[int](#int)</code> | Number of states processed in a single job. | <code>10000</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`initial` | <code>[Contexts](#edmkit.search.energy.energy.Contexts)</code> | Initial context of shape ``(1, 0)``.
`plan` | <code>[Plan](#edmkit.search.energy.energy.Plan)</code> | Plan that yields one job per ``batch_size`` chunk of states.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | If ``theiler_window`` is negative.

