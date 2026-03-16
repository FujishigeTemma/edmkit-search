import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from edmkit.search.metrics import (
    PER_DIM_METRICS,
    SCALAR_METRICS,
    mae,
    mae_per_dim,
    mean_rho,
    mean_rho_per_dim,
    rmse,
    rmse_per_dim,
)
from tests.strategies import arrays_2d, matched_arrays, reasonable_floats

ERROR_PER_DIM = [rmse_per_dim, mae_per_dim]
ERROR_SCALAR = [rmse, mae]
ALL_METRICS = SCALAR_METRICS + PER_DIM_METRICS


# ---------------------------------------------------------------------------
# 1. Validation — shape mismatch
# ---------------------------------------------------------------------------
@given(
    n=st.integers(min_value=2, max_value=50),
    m1=st.integers(min_value=1, max_value=5),
    m2=st.integers(min_value=1, max_value=5),
)
@pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda m: m.__name__)
def test_shape_mismatch_raises(metric, n, m1, m2):
    """All metrics reject (N, M1) vs (N, M2) when M1 != M2."""
    assume(m1 != m2)
    predictions = np.zeros((n, m1))
    observations = np.zeros((n, m2))
    with pytest.raises(ValueError, match="Shape mismatch"):
        metric(predictions, observations)


# ---------------------------------------------------------------------------
# 2. Validation — 1D input
# ---------------------------------------------------------------------------
@given(n=st.integers(min_value=2, max_value=100))
@pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda m: m.__name__)
def test_1d_input_raises(metric, n):
    """All metrics reject 1D arrays."""
    arr = np.zeros(n)
    with pytest.raises(ValueError, match="2D"):
        metric(arr, arr)


# ---------------------------------------------------------------------------
# 3. Invariant (structural) — per-dim output shape
# ---------------------------------------------------------------------------
@given(data=matched_arrays())
@pytest.mark.parametrize("metric", PER_DIM_METRICS, ids=lambda m: m.__name__)
def test_per_dim_output_shape(metric, data):
    """Per-dim metrics return shape (M,)."""
    predictions, observations = data
    result = metric(predictions, observations)
    assert result.shape == (predictions.shape[1],)


# ---------------------------------------------------------------------------
# 4. Invariant (range) — mean_rho_per_dim bounded
# ---------------------------------------------------------------------------
@given(data=matched_arrays(min_n=3))
def test_mean_rho_per_dim_bounded(data):
    """mean_rho_per_dim values lie in [-1, 1] (with tolerance for floating-point)."""
    predictions, observations = data
    result = mean_rho_per_dim(predictions, observations)
    assert np.all(result >= -1 - 1e-5)
    assert np.all(result <= 1 + 1e-5)


# ---------------------------------------------------------------------------
# 5. Algebraic (identity) — self-correlation
# ---------------------------------------------------------------------------
@given(arr=arrays_2d(min_n=3))
def test_self_correlation_per_dim(arr):
    """mean_rho_per_dim(x, x) == 1.0 for non-constant columns."""
    assume(np.all(arr.std(axis=0) > 1e-6))
    result = mean_rho_per_dim(arr, arr)
    np.testing.assert_allclose(result, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 6. Robustness — mean_rho_per_dim always finite
# ---------------------------------------------------------------------------
@given(data=matched_arrays(min_n=3))
def test_mean_rho_per_dim_finite(data):
    """mean_rho_per_dim output is always finite (no NaN/Inf)."""
    predictions, observations = data
    result = mean_rho_per_dim(predictions, observations)
    assert np.all(np.isfinite(result))


def test_mean_rho_per_dim_constant_column():
    """mean_rho_per_dim returns 0 for constant columns (zero variance)."""
    predictions = np.array([[1.0, 2.0], [1.0, 3.0]])
    observations = np.array([[5.0, 1.0], [5.0, 4.0]])
    result = mean_rho_per_dim(predictions, observations)
    assert result[0] == 0.0
    assert np.isfinite(result[1])


# ---------------------------------------------------------------------------
# 7. Algebraic (commutativity) — mean_rho_per_dim symmetric
# ---------------------------------------------------------------------------
@given(data=matched_arrays(min_n=3))
def test_mean_rho_per_dim_symmetric(data):
    """mean_rho_per_dim(a, b) == mean_rho_per_dim(b, a)."""
    a, b = data
    np.testing.assert_allclose(
        mean_rho_per_dim(a, b), mean_rho_per_dim(b, a), rtol=1e-10, atol=1e-12
    )


# ---------------------------------------------------------------------------
# 8. Metamorphic — mean_rho_per_dim shift-invariant
# ---------------------------------------------------------------------------
@given(data=matched_arrays(min_n=3), c=reasonable_floats)
def test_mean_rho_per_dim_shift_invariant(data, c):
    """mean_rho_per_dim(predictions + c, observations) == mean_rho_per_dim(predictions, observations)."""
    predictions, observations = data
    min_std = predictions.std(axis=0).min()
    assume(min_std > 1e-6)
    assume(np.all(observations.std(axis=0) > 1e-6))
    # Centering loses ~eps * |c/std| digits of precision.
    condition = abs(c) / min_std
    rtol = max(1e-7, condition * np.finfo(float).eps * 10)
    np.testing.assert_allclose(
        mean_rho_per_dim(predictions + c, observations),
        mean_rho_per_dim(predictions, observations),
        rtol=rtol,
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# 9. Metamorphic — error per-dim metrics scale-equivariant
# ---------------------------------------------------------------------------
@given(data=matched_arrays(), a=reasonable_floats)
@pytest.mark.parametrize("metric", ERROR_PER_DIM, ids=lambda m: m.__name__)
def test_error_per_dim_scale_equivariant(metric, data, a):
    """f(a*p, a*o) = |a|*f(p, o)."""
    assume(a != 0.0)
    predictions, observations = data
    np.testing.assert_allclose(
        metric(a * predictions, a * observations),
        np.abs(a) * metric(predictions, observations),
        rtol=1e-7,
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# 10. Scalar output type
# ---------------------------------------------------------------------------
@given(data=matched_arrays())
@pytest.mark.parametrize("metric", SCALAR_METRICS, ids=lambda m: m.__name__)
def test_scalar_returns_float(metric, data):
    """Scalar metrics return a Python float."""
    predictions, observations = data
    result = metric(predictions, observations)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# 11. Scalar mean_rho consistency with per-dim
# ---------------------------------------------------------------------------
@given(data=matched_arrays(min_n=3))
def test_mean_rho_equals_mean_of_per_dim(data):
    """mean_rho(p, o) == mean(mean_rho_per_dim(p, o))."""
    predictions, observations = data
    np.testing.assert_allclose(
        mean_rho(predictions, observations),
        float(np.mean(mean_rho_per_dim(predictions, observations))),
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# 12. Scalar mae consistency with per-dim
# ---------------------------------------------------------------------------
@given(data=matched_arrays())
def test_mae_equals_mean_of_per_dim(data):
    """mae(p, o) == mean(mae_per_dim(p, o))."""
    predictions, observations = data
    np.testing.assert_allclose(
        mae(predictions, observations),
        float(np.mean(mae_per_dim(predictions, observations))),
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# 13. Scalar rmse definition
# ---------------------------------------------------------------------------
@given(data=matched_arrays())
def test_rmse_definition(data):
    """rmse(p, o) == sqrt(mean((p - o)^2)) over all elements."""
    predictions, observations = data
    expected = float(np.sqrt(np.mean((predictions - observations) ** 2)))
    np.testing.assert_allclose(rmse(predictions, observations), expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# 14. Scalar error metrics scale-equivariant
# ---------------------------------------------------------------------------
@given(data=matched_arrays(), a=reasonable_floats)
@pytest.mark.parametrize("metric", ERROR_SCALAR, ids=lambda m: m.__name__)
def test_error_scalar_scale_equivariant(metric, data, a):
    """f(a*p, a*o) = |a|*f(p, o) for scalar error metrics."""
    assume(a != 0.0)
    predictions, observations = data
    np.testing.assert_allclose(
        metric(a * predictions, a * observations),
        abs(a) * metric(predictions, observations),
        rtol=1e-7,
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# 15. Scalar mean_rho bounded
# ---------------------------------------------------------------------------
@given(data=matched_arrays(min_n=3))
def test_mean_rho_scalar_bounded(data):
    """Scalar mean_rho lies in [-1, 1]."""
    predictions, observations = data
    result = mean_rho(predictions, observations)
    assert result >= -1 - 1e-5
    assert result <= 1 + 1e-5


# ---------------------------------------------------------------------------
# 16. Scalar self-correlation
# ---------------------------------------------------------------------------
@given(arr=arrays_2d(min_n=3))
def test_self_correlation_scalar(arr):
    """mean_rho(x, x) == 1.0 for non-constant columns."""
    assume(np.all(arr.std(axis=0) > 1e-6))
    np.testing.assert_allclose(mean_rho(arr, arr), 1.0, atol=1e-10)
