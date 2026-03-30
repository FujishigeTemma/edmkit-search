import hashlib
import threading

import numpy as np
from edmkit.ccm import bootstrap
from edmkit.embedding import lagged_embed, scan, select

from edmkit.types import PredictFunc

from .types import FilterFn


def causation(
    X: np.ndarray,
    Y: np.ndarray,
    lib_sizes: list[int],
    *,
    predict: PredictFunc,
    E: int,
    tau: int,
    n_samples: int = 20,
    slope_threshold: float = 0.002,
    rng: np.random.Generator | None = None,
) -> bool:
    """Test for CCM convergence using slope of ρ vs library size.

    Embeds Y (effect), predicts X (cause) via bootstrap CCM at multiple
    library sizes, then fits a linear regression of mean ρ against
    normalized library size.  Returns ``True`` if the slope exceeds the
    threshold, indicating that prediction skill increases with library
    size — the hallmark of convergent cross-mapping.

    Parameters
    ----------
    X : np.ndarray of shape (T,) or (T, M)
        Potential cause variable(s).
    Y : np.ndarray of shape (T,) or (T, M)
        Potential effect variable(s).
    lib_sizes : list[int]
        Library sizes to test (e.g. [100, 200, 500, 1000, 2000, 5000]).
    predict : PredictFunc
        Prediction function ``(X, Y, Q, *, mask) -> predictions``.
    E : int
        Embedding dimension.
    tau : int
        Time delay for embedding.
    n_samples : int
        Number of bootstrap samples to collect. Default is 20.
    slope_threshold : float
        Minimum slope of mean ρ vs normalized library size to declare
        convergence.  Default is 0.002 (matches MDE/dimx).
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    bool
        True if convergence is detected (slope >= threshold).

    Raises
    ------
    ValueError
        If `lib_sizes` is empty or contains non-positive values,
        or if `n_samples`, `E`, or `tau` is not positive.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if not lib_sizes:
        raise ValueError("lib_sizes cannot be empty")
    if any(s <= 0 for s in lib_sizes):
        raise ValueError(f"All lib_sizes must be positive, got {lib_sizes}")
    if E <= 0:
        raise ValueError(f"E must be positive, got {E}")
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    if rng is None:
        rng = np.random.default_rng()

    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    # To test X -> Y: embed Y (effect), predict X (cause) from Y's attractor
    M = Y.shape[1]
    Y_embedded_list = []
    min_L = Y.shape[0]
    for m in range(M):
        embedded = lagged_embed(Y[:, m], tau, E)
        Y_embedded_list.append(embedded)
        min_L = min(min_L, embedded.shape[0])

    offset = Y.shape[0] - min_L
    Y_embedded = np.concatenate(
        [e[-min_L:] for e in Y_embedded_list], axis=1
    )  # (L, M*E)
    X_aligned = X[offset:]  # (L, M_x)

    L = Y_embedded.shape[0]
    library_pool = np.arange(L)
    prediction_pool = np.arange(L)

    def sampler(pool: np.ndarray, size: int) -> np.ndarray:
        return rng.choice(pool, size=size, replace=True)

    samples = bootstrap(
        Y_embedded,
        X_aligned,
        np.array(lib_sizes, dtype=int),
        predict_func=predict,
        n_samples=n_samples,
        library_pool=library_pool,
        prediction_pool=prediction_pool,
        sample_func=sampler,
    )  # (n_samples, len(lib_sizes))

    means = samples.mean(axis=0)  # (len(lib_sizes),)

    # Slope of mean ρ vs normalized library size via OLS
    x = np.array(lib_sizes, dtype=float)
    x = x / x.max()  # normalize to [0, 1]
    x_c = x - x.mean()
    y_c = means - means.mean()
    denom = x_c @ x_c
    if denom == 0:
        return False
    slope = float(x_c @ y_c / denom)

    return slope >= slope_threshold


def make_ccm_filter(
    *,
    predict: PredictFunc,
    E: list[int],
    tau: list[int],
    lib_sizes: list[int],
    n_samples: int = 20,
    slope_threshold: float = 0.002,
    rho_min: float = 0.0,
    seed: int = 0,
) -> FilterFn:
    """Create a CCM convergence filter with dynamic E/tau estimation.

    Two-layer filtering:

    1. Run ``scan`` + ``select`` to find the best (E, tau) for the
       candidate variable.  If the mean score is below ``rho_min``, the
       variable is rejected immediately (no detectable dynamics).
    2. Run ``causation`` with the selected (E, tau) to test for CCM
       convergence.

    Results are cached per column (thread-safe) so repeated calls with
    the same ``x`` array are free.

    Parameters
    ----------
    predict : PredictFunc
        Prediction function for both scan and CCM.
    E : list[int]
        Embedding dimension candidates.
    tau : list[int]
        Time delay candidates.
    lib_sizes : list[int]
        Library sizes for CCM convergence testing.
    n_samples : int
        Bootstrap samples for convergence test.
    slope_threshold : float
        Minimum slope to declare convergence (passed to ``causation``).
    rho_min : float
        Minimum mean score from ``select`` to proceed with CCM.
        Set to 0.0 (default) to disable early rejection.
    seed : int
        Base seed for deterministic per-column RNG.

    Returns
    -------
    FilterFn
        Filter function ``(x, Y) -> bool``.
    """
    cache: dict[bytes, bool] = {}
    lock = threading.Lock()

    def ccm_filter(x: np.ndarray, Y: np.ndarray) -> bool:
        key = x.tobytes()
        with lock:
            if key in cache:
                return cache[key]

        scores = scan(x, E=E, tau=tau, predict=predict)
        best_E, best_tau, best_score = select(scores, E=E, tau=tau)

        if best_score < rho_min:
            with lock:
                cache[key] = False
            return False

        col_hash = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
        rng = np.random.default_rng([seed, col_hash])
        result = causation(
            x,
            Y,
            lib_sizes,
            predict=predict,
            E=best_E,
            tau=best_tau,
            n_samples=n_samples,
            slope_threshold=slope_threshold,
            rng=rng,
        )

        with lock:
            cache[key] = result
        return result

    return ccm_filter
