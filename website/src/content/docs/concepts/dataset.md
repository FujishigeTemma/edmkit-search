---
title: Dataset
description: Time-series containers and on-the-fly transforms.
---

A `Dataset` is the input arm of a search. It holds an `(T, D_x)` input array `X`, a target array `Y`, and an optional `Transform` applied lazily at sample-access time.

## Basics

```python
from edmkit.search.dataset import Dataset

data = Dataset(X=X_full, Y=Y_full)
x, y = data[0]            # one sample
n   = len(data)           # T
```

`X` must be 2D. `Y` may be 1D or 2D; a 1D `Y` is auto-promoted to `(T, 1)` so downstream code can treat the target as uniformly 2D. Both arrays are cast to `float32` on construction.

## Subsets

A `Subset` is a non-copying view into a `Dataset` restricted to a 1D integer index array. It *is* a `Dataset` (subclass), so it can be passed wherever a dataset is expected — for example, as the train or validation arm of a `Fold`:

```python
from edmkit.search.dataset import Subset
from edmkit.splits import temporal_fold

outer = temporal_fold(len(data), train_ratio=0.8)
train      = Subset(data, outer.train)
validation = Subset(data, outer.validation)
```

`Subset.X` and `Subset.Y` are materialized lazily via `@cached_property`, so a subset that is never accessed costs only the index array.

## Transforms

A `Transform` is just a closure `(x, y) -> (x', y')`. It is applied per-sample at `__getitem__` time, so transforms with internal randomness (data augmentation) see fresh noise on every pass.

Two built-in factories are provided:

```python
from edmkit.search.dataset import compose, gaussian_noise, zscore_normalize

normalize = zscore_normalize(X_train, target="x")  # closes over train-set stats
augment   = gaussian_noise(sigma=0.05)
transform = compose(normalize, augment)            # left-to-right

data = Dataset(X, Y, transform=transform)
```

- `zscore_normalize(data, target=...)` computes mean and standard deviation once over the leading axes of `data` (`(T, D)` or `(N, T, D)`) and bakes them into the closure. `target` is `"x"`, `"y"`, or `"both"`.
- `gaussian_noise(sigma, rng)` perturbs only the input arm. Pass a seeded `Generator` for reproducibility.
- `compose(*transforms)` left-folds — the first transform's output feeds the second, and so on.

## Why a Class and Not Just Arrays?

The bare arrays `X` and `Y` would have worked. The `Dataset` wrapper exists to:

1. Validate shape contracts once, at construction time.
2. Give `Subset` somewhere to live so train/validation arms can share the underlying storage by reference.
3. Provide a hook (`transform`) that energies can rely on for on-the-fly augmentation without re-implementing it everywhere.

If you do not need a transform, treating `Dataset` as a thin shape-checked pair is fine.
