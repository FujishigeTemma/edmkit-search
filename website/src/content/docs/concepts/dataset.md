---
title: Dataset
description: The (X, Y) container, the forecasting-horizon convention, and non-copying subsets.
---

A `Dataset` holds an `(T, D_x)` input array `X`, a target array `Y` aligned along the same time axis, and an optional `Transform` applied lazily at sample access. A `Subset` is a non-copying view used to carve fold arms — train and validation arms share underlying storage, so two-level folds cost essentially nothing.

If you don't need a transform, treat `Dataset` as a thin shape-checked pair.

## Basic usage

```python
import numpy as np
from edmkit.search.dataset import Dataset

X = np.random.default_rng(0).standard_normal((1000, 8))   # T=1000, D_x=8
Y = X[:, :1]                                              # T=1000, D_y=1

data = Dataset(X=X, Y=Y)
x, y = data[0]            # one sample, transform applied if any
n   = len(data)           # T = 1000
```

`X` must be 2D `(T, D_x)`. `Y` may be 1D `(T,)` or 2D `(T, D_y)`; a 1D `Y` is auto-promoted. Both arrays are cast to `float32` on construction. Mismatched shapes raise `ValueError`.

## The forecasting-horizon shift

To predict `Y` at `t + n_ahead` from `X` at `t`, shift the arrays before constructing the `Dataset`:

```python
n_ahead = 1
T = X_full.shape[0]
data = Dataset(X=X_full[:T - n_ahead], Y=Y_full[n_ahead:])
```

After the shift, `data.X[t]` and `data.Y[t]` are `X_full[t]` and `Y_full[t + n_ahead]`. The dataset length is `T - n_ahead`. For nowcasting, pass `X_full` and `Y_full` directly.

:::caution
Apply the shift **before** any fold split. Splitting first and then shifting inside each arm leaks future information across the fold boundary.
:::

## Subsets are fold arms

A `Subset` is a non-copying view restricted to a 1D integer index array. It *is* a `Dataset` (subclass), so it can be passed wherever a dataset is expected:

```python
from edmkit.search.dataset import Subset
from edmkit.splits import temporal_fold

outer = temporal_fold(data.X.shape[0], 0.8)
train      = Subset(data, outer.train)
validation = Subset(data, outer.validation)
```

`Subset.X` and `Subset.Y` are materialized lazily via `@cached_property`, so an unused subset costs only the index array. This makes the [two-level fold pattern](/edmkit-search/concepts/validation/) cheap.

[`edmkit.splits`](https://fujishigetemma.github.io/edmkit/reference/splits/) provides the index generators:

- `temporal_fold(n, train_ratio)` — one chronological split.
- `sliding_folds(n, train_size, validation_size, stride)` — a sequence of fixed-size windows, used by `energy.cross.folds`.
- `expanding_folds(...)` — a sequence with a growing train window.

## Transforms

A `Transform` is a closure `(x, y) -> (x', y')` applied at `__getitem__`. The shipped transforms are `zscore_normalize`, `gaussian_noise`, and `compose`.

:::note[Transforms do not run during scoring]
The built-in energies read `Dataset.X` and `Dataset.Y` directly, so transforms affect only downstream code that iterates sample by sample. To affect the energy, apply normalization before constructing the `Dataset`.
:::

