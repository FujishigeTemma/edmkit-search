from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from edmkit.search.dataset import Dataset, Subset, Transform, compose, gaussian_noise, zscore_normalize


class DatasetCase(NamedTuple):
    X: np.ndarray
    Y: np.ndarray
    transform: Transform | None = None


class SubsetCase(NamedTuple):
    indices: np.ndarray


class ZScoreProblem(NamedTuple):
    data: np.ndarray
    x: np.ndarray
    y: np.ndarray
    target: str


def check_dataset(X: np.ndarray, Y: np.ndarray, transform: Transform | None = None) -> None:
    data = Dataset(X, Y, transform=transform)
    promoted = Y[:, None] if Y.ndim == 1 else Y

    assert len(data) == X.shape[0]
    assert data.X.dtype == np.float32 and data.Y.dtype == np.float32
    np.testing.assert_allclose(data.X, X, rtol=1e-6)  # stored arrays stay untransformed
    np.testing.assert_allclose(data.Y, promoted, rtol=1e-6)
    for i in range(len(data)):
        expected = (data.X[i], data.Y[i]) if transform is None else transform(data.X[i], data.Y[i])
        x, y = data[i]
        np.testing.assert_array_equal(x, expected[0])
        np.testing.assert_array_equal(y, expected[1])


def check_subset(indices: np.ndarray) -> None:
    X = np.arange(12.0).reshape(6, 2)
    data = Dataset(X, X[:, :1], transform=lambda x, y: (x * 2, y))
    subset = Subset(data, indices)

    assert isinstance(subset, Dataset)
    assert len(subset) == len(indices)
    np.testing.assert_array_equal(subset.X, data.X[indices])
    np.testing.assert_array_equal(subset.Y, data.Y[indices])
    for i, index in enumerate(indices.tolist()):
        x, y = subset[i]
        expected_x, expected_y = data[index]  # delegation includes the transform, on both arms
        np.testing.assert_array_equal(x, expected_x)
        np.testing.assert_array_equal(y, expected_y)


def zscore_reference(data: np.ndarray, values: np.ndarray) -> np.ndarray:
    flat = data.reshape(-1, data.shape[-1])
    return (values - flat.mean(axis=0)) / flat.std(axis=0)


def check_zscore(data: np.ndarray, x: np.ndarray, y: np.ndarray, target: str) -> None:
    normalized_x, normalized_y = zscore_normalize(data, target=target)(x, y)

    expected_x = zscore_reference(data, x) if target in ("x", "both") else x
    expected_y = zscore_reference(data, y) if target in ("y", "both") else y
    np.testing.assert_allclose(normalized_x, expected_x, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(normalized_y, expected_y, rtol=1e-4, atol=1e-5)


@st.composite
def zscore_problems(draw):
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    columns = draw(st.integers(1, 4))
    rows = draw(st.integers(3, 20))
    # Statistics flatten every leading axis, so 2D and 3D reference data are equivalent shapes.
    shape = (rows, columns) if draw(st.booleans()) else (draw(st.integers(1, 3)), rows, columns)
    # Normal draws keep the std safely away from the implementation's zero-variance guard.
    return ZScoreProblem(rng.normal(size=shape), rng.normal(size=columns), rng.normal(size=columns), draw(st.sampled_from(["x", "y", "both"])))


X5 = np.arange(15.0).reshape(5, 3)

DATASET_VALID = {
    "2d-target": DatasetCase(X5, np.arange(10.0).reshape(5, 2)),
    "1d-target-promoted": DatasetCase(X5, np.arange(5.0)),
    "transform-applied-at-access": DatasetCase(X5, np.arange(5.0), lambda x, y: (x * 2, y + 1)),
}

DATASET_INVALID = {
    "1d-X": DatasetCase(np.zeros(5), np.zeros(5)),
    "3d-X": DatasetCase(np.zeros((5, 3, 2)), np.zeros(5)),
    "3d-Y": DatasetCase(X5, np.zeros((5, 2, 2))),
    "length-mismatch": DatasetCase(X5, np.zeros((4, 2))),
}

SUBSET_VALID = {
    "reordered": SubsetCase(np.array([4, 1, 3])),
    "single-row": SubsetCase(np.array([2])),
    "empty": SubsetCase(np.array([], dtype=np.int64)),
}


@pytest.mark.parametrize("case", DATASET_VALID.values(), ids=DATASET_VALID.keys())
def test_dataset_valid(case: DatasetCase) -> None:
    check_dataset(*case)


@pytest.mark.parametrize("case", DATASET_INVALID.values(), ids=DATASET_INVALID.keys())
def test_dataset_invalid(case: DatasetCase) -> None:
    with pytest.raises(ValueError):
        Dataset(case.X, case.Y)


@pytest.mark.parametrize("case", SUBSET_VALID.values(), ids=SUBSET_VALID.keys())
def test_subset_valid(case: SubsetCase) -> None:
    check_subset(*case)


@given(problem=zscore_problems())
def test_zscore_normalize_compatibility(problem: ZScoreProblem) -> None:
    check_zscore(*problem)


def test_zscore_normalize_constant_feature() -> None:
    # Zero variance has no well-defined reference — the documented guard just has to keep the output finite.
    x, _ = zscore_normalize(np.ones((4, 2)), target="x")(np.ones(2), np.ones(2))

    assert np.isfinite(x).all()


@pytest.mark.parametrize("target", ["z", ""], ids=["unknown", "empty"])
def test_zscore_normalize_invalid(target: str) -> None:
    with pytest.raises(ValueError, match="target"):
        zscore_normalize(np.ones((3, 2)), target=target)


def test_gaussian_noise() -> None:
    # A trivial closure — one example test: same rng seed reproduces the noise, y passes through, x actually changes.
    x, y = np.zeros(4), np.ones(2)

    first = gaussian_noise(sigma=0.5, rng=np.random.default_rng(0))(x, y)
    second = gaussian_noise(sigma=0.5, rng=np.random.default_rng(0))(x, y)

    np.testing.assert_array_equal(first[0], second[0])
    assert not np.array_equal(first[0], x)
    np.testing.assert_array_equal(first[1], y)


def test_compose() -> None:
    add_one = lambda x, y: (x + 1, y)  # noqa: E731
    double = lambda x, y: (x * 2, y)  # noqa: E731

    x, _ = compose(add_one, double)(np.array([1.0]), np.array([0.0]))

    np.testing.assert_array_equal(x, np.array([4.0]))  # (1 + 1) * 2 — left to right, not 1 * 2 + 1
