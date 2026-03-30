"""Tests for edmkit.search.convergence (causation + make_ccm_filter)."""

import numpy as np
from edmkit.simplex_projection import simplex_projection

from edmkit.search.convergence import make_ccm_filter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CAUSAL_PAIR_CACHE: tuple[np.ndarray, np.ndarray] | None = None


def _causal_pair() -> tuple[np.ndarray, np.ndarray]:
    """Coupled logistic maps with one-way coupling X -> Y."""
    global _CAUSAL_PAIR_CACHE
    if _CAUSAL_PAIR_CACHE is not None:
        return _CAUSAL_PAIR_CACHE
    n_total = 1050
    rx, ry, bxy = 3.8, 3.5, 0.02
    x = np.zeros(n_total)
    y = np.zeros(n_total)
    x[0], y[0] = 0.4, 0.2
    for i in range(1, n_total):
        x[i] = x[i - 1] * (rx - rx * x[i - 1])
        y[i] = y[i - 1] * (ry - ry * y[i - 1]) + bxy * x[i - 1]
    _CAUSAL_PAIR_CACHE = (x[50:], y[50:])
    return _CAUSAL_PAIR_CACHE


# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

_E = [2, 3, 4]
_TAU = [1, 2]
_LIB_SIZES = [100, 200, 500]
_N_SAMPLES = 10
_SEED = 42


def _make_filter(*, rho_min: float = 0.0):  # -> FilterFn
    return make_ccm_filter(
        predict=simplex_projection,
        E=_E,
        tau=_TAU,
        lib_sizes=_LIB_SIZES,
        n_samples=_N_SAMPLES,
        seed=_SEED,
        rho_min=rho_min,
    )


# ---------------------------------------------------------------------------
# make_ccm_filter
# ---------------------------------------------------------------------------


class TestMakeCcmFilter:
    def test_returns_callable_with_name(self):
        f = _make_filter()
        assert callable(f)
        assert hasattr(f, "__name__")

    def test_causal_pair_accepted(self):
        x, y = _causal_pair()
        f = _make_filter()
        assert f(x, y[:, None]) is True

    def test_caches_results(self):
        x, y = _causal_pair()
        f = _make_filter()
        result1 = f(x, y[:, None])
        result2 = f(x, y[:, None])
        assert result1 == result2

    def test_rho_min_rejects_noise(self):
        """Noise has near-zero self-prediction score; rho_min catches it
        before the expensive CCM test even runs."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(500)
        y = rng.standard_normal(500)
        f = _make_filter(rho_min=0.3)
        assert f(x, y[:, None]) is False

    def test_rho_min_does_not_block_causal_pair(self):
        """Causal pair has high self-prediction score and passes rho_min."""
        x, y = _causal_pair()
        f = _make_filter(rho_min=0.3)
        assert f(x, y[:, None]) is True
