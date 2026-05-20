---
title: dataset
description: Time-series dataset containers and transforms.
sidebar:
  order: 1
---

## `dataset`

Dataset subpackage: containers and transforms.

**Modules:**

Name | Description
---- | -----------
[`containers`](#edmkit.search.dataset.containers) | 
[`transforms`](#edmkit.search.dataset.transforms) | 

### `transforms`

**Functions:**

Name | Description
---- | -----------
[`zscore_normalize`](#edmkit.search.dataset.transforms.zscore_normalize) | Build a `Transform` that z-score normalizes using statistics from ``data``.
[`gaussian_noise`](#edmkit.search.dataset.transforms.gaussian_noise) | Build a `Transform` that perturbs the input with Gaussian noise.
[`compose`](#edmkit.search.dataset.transforms.compose) | Compose multiple transforms into a single left-to-right pipeline.

#### `zscore_normalize`

```python
zscore_normalize(data: np.ndarray, *, target: str) -> Transform
```

Build a `Transform` that z-score normalizes using statistics from ``data``.

The mean and standard deviation are computed once over the leading
axes of ``data`` (treating the last axis as the feature axis) and
captured in the returned closure. The closure can then be applied
repeatedly to individual samples without recomputing statistics.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>np.ndarray of shape (T, D) or (N, T, D)</code> | Reference data from which the per-feature mean and standard deviation are computed. The last axis is treated as the feature axis. | *required*
`target` | <code>([x](#x), [y](#y), [both](#both))</code> | Which arm of the ``(x, y)`` pair to normalize. ``"both"`` is only valid when ``D_x == D_y`` (the same statistics are used for both arms). | <code>"x"</code>

**Returns:**

Type | Description
---- | -----------
<code>[Transform](#edmkit.search.dataset.transforms.Transform)</code> | ``(x, y) -> (x', y')`` where the selected arm(s) are normalized.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | If ``target`` is not one of ``"x"``, ``"y"``, ``"both"``.

**Examples:**

```python
zscore_x = zscore_normalize(X_train, target="x")
zscore_y = zscore_normalize(Y_train, target="y")
transform = compose(zscore_x, zscore_y)  # independent stats per arm
```

#### `gaussian_noise`

```python
gaussian_noise(sigma: float = 0.1, rng: np.random.Generator | None = None) -> Transform
```

Build a `Transform` that perturbs the input with Gaussian noise.

The noise is drawn at sample-access time, so each pass through
the dataset sees fresh noise — making this suitable for
on-the-fly data augmentation. Only the input arm ``x`` is
perturbed; ``y`` is returned unchanged.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`sigma` | <code>[float](#float)</code> | Standard deviation of the noise. | <code>0.1</code>
`rng` | <code>[Generator](#numpy.random.Generator) or None</code> | Random number generator for reproducibility. When ``None``, a fresh unseeded generator is created. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[Transform](#edmkit.search.dataset.transforms.Transform)</code> | ``(x, y) -> (x + noise, y)``.

#### `compose`

```python
compose(*transforms: Transform) -> Transform
```

Compose multiple transforms into a single left-to-right pipeline.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`*transforms` | <code>[Transform](#edmkit.search.dataset.transforms.Transform)</code> | Transforms to apply in order. The output of each transform feeds the next. | <code>()</code>

**Returns:**

Type | Description
---- | -----------
<code>[Transform](#edmkit.search.dataset.transforms.Transform)</code> | ``(x, y) -> transforms[-1](... transforms[1](transforms[0](x, y)))``.

**Examples:**

```python
transform = compose(zscore_normalize(X_train, target="x"), gaussian_noise(0.05))
# z-score is applied first, then noise is added on top
```

### `containers`

**Classes:**

Name | Description
---- | -----------
[`Dataset`](#edmkit.search.dataset.containers.Dataset) | Time-series dataset of paired input/output sequences.
[`Subset`](#edmkit.search.dataset.containers.Subset) | A non-copying view into a `Dataset` restricted to selected rows.

#### `Dataset`

```python
Dataset(X: np.ndarray, Y: np.ndarray, *, transform: Transform | None = None)
```

Time-series dataset of paired input/output sequences.

Holds an input array ``X`` and a target array ``Y`` sharing a common
time axis, plus an optional `Transform` applied lazily at
``__getitem__`` time. Inputs are cast to ``float32`` and a 1D ``Y``
is auto-promoted to ``(T, 1)`` so downstream code can treat the
target as 2D uniformly.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>np.ndarray of shape (T, D_x)</code> | Input time series. | *required*
`Y` | <code>np.ndarray of shape (T, D_y) or (T,)</code> | Target time series. 1D input is promoted to ``(T, 1)``. | *required*
`transform` | <code>[Transform](#edmkit.search.dataset.transforms.Transform) or None</code> | ``(x, y) -> (x', y')`` closure applied at indexing time. Used for preprocessing or on-the-fly data augmentation. | <code>None</code>

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | - If ``X`` is not 2-dimensional. - If ``Y`` is not 1D or 2D. - If ``X`` and ``Y`` have different lengths along the time axis.

**Examples:**

```python
import numpy as np

from edmkit.search.dataset import Dataset, zscore_normalize

X = np.random.default_rng(0).standard_normal((1000, 8))
Y = X[:, :1]

data = Dataset(X, Y, transform=zscore_normalize(X, target="x"))
x, y = data[0]
```

#### `Subset`

```python
Subset(dataset: Dataset, indices: np.ndarray)
```

Bases: <code>[Dataset](#edmkit.search.dataset.containers.Dataset)</code>

A non-copying view into a `Dataset` restricted to selected rows.

``Subset`` is a ``Dataset`` subtype, so it can be passed anywhere
a dataset is expected (e.g. as the train/validation arms of a
fold). ``X`` and ``Y`` are materialized lazily via
``cached_property``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dataset` | <code>[Dataset](#edmkit.search.dataset.containers.Dataset)</code> | Underlying dataset. | *required*
`indices` | <code>[ndarray](#numpy.ndarray)</code> | 1D integer indices selecting rows of ``dataset`` to expose. The exposed length is ``len(indices)``. | *required*

**Examples:**

```python
import numpy as np

from edmkit.search.dataset import Dataset, Subset

data = Dataset(X, Y)
train = Subset(data, np.arange(800))
validation = Subset(data, np.arange(800, 1000))
```

