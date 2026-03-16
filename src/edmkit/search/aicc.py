import numpy as np
from scipy.optimize import least_squares


def fit_linear(
    L: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float]:
    """Fit weighted linear model s(L) = alpha + beta * L.

    Parameters
    ----------
    L : np.ndarray of shape (n,)
        Library sizes (independent variable).
    scores : np.ndarray of shape (n,)
        Observed scores at each library size.
    weights : np.ndarray of shape (n,)
        Non-negative weights for each observation.

    Returns
    -------
    tuple[float, float, float]
        (alpha, beta, rss) — intercept, slope, and residual sum of squares.

    Raises
    ------
    ValueError
        If inputs are not 1D or have mismatched shapes.
    """
    if L.ndim != 1 or scores.ndim != 1 or weights.ndim != 1:
        raise ValueError(
            f"All inputs must be 1D, got L={L.ndim}D, scores={scores.ndim}D, weights={weights.ndim}D"
        )
    if L.shape != scores.shape or L.shape != weights.shape:
        raise ValueError(
            f"Shape mismatch: L={L.shape}, scores={scores.shape}, weights={weights.shape}"
        )
    A = np.column_stack([np.ones_like(L), L])
    Aw = A * weights[:, None]
    alpha, beta = np.linalg.solve(Aw.T @ A, Aw.T @ scores)
    residuals = scores - (alpha + beta * L)
    rss = float(np.sum(residuals**2))
    return alpha, beta, rss


def fit_saturation(
    L: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float, float]:
    """Fit weighted saturation model s(L) = a - b * exp(-c * L).

    b and c are constrained positive via exp reparameterization.

    Parameters
    ----------
    L : np.ndarray of shape (n,)
        Library sizes (independent variable).
    scores : np.ndarray of shape (n,)
        Observed scores at each library size.
    weights : np.ndarray of shape (n,)
        Non-negative weights for each observation.

    Returns
    -------
    tuple[float, float, float, float]
        (a, b, c, rss) — asymptote, amplitude, rate, and residual sum of squares.

    Raises
    ------
    ValueError
        If inputs are not 1D or have mismatched shapes.
    """
    if L.ndim != 1 or scores.ndim != 1 or weights.ndim != 1:
        raise ValueError(
            f"All inputs must be 1D, got L={L.ndim}D, scores={scores.ndim}D, weights={weights.ndim}D"
        )
    if L.shape != scores.shape or L.shape != weights.shape:
        raise ValueError(
            f"Shape mismatch: L={L.shape}, scores={scores.shape}, weights={weights.shape}"
        )
    a0 = float(np.max(scores))
    b0 = max(a0 - float(scores[0]), 1e-6)
    L_mid = float(np.median(L))
    c0 = max(np.log(2) / L_mid if L_mid > 0 else 0.01, 1e-6)

    x0 = np.array([a0, np.log(b0), np.log(c0)])
    sqrt_w = np.sqrt(weights)

    def residual_fn(x: np.ndarray) -> np.ndarray:
        a, b_raw, c_raw = x
        b = np.exp(b_raw)
        c = np.exp(c_raw)
        predicted = a - b * np.exp(-c * L)
        return sqrt_w * (scores - predicted)

    result = least_squares(residual_fn, x0, method="lm", max_nfev=2000)
    a = result.x[0]
    b = np.exp(result.x[1])
    c = np.exp(result.x[2])
    residuals = scores - (a - b * np.exp(-c * L))
    rss = float(np.sum(residuals**2))
    return a, b, c, rss


def aicc(rss: float, n: int, k: int) -> float:
    """Compute AICc (or AIC fallback when n <= k + 1).

    Parameters
    ----------
    rss : float
        Residual sum of squares.
    n : int
        Number of observations.
    k : int
        Number of model parameters.

    Returns
    -------
    float
        AICc value (with small-sample correction when ``n > k + 1``).

    Raises
    ------
    ValueError
        If `n` or `k` are not positive, or `rss` is negative.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if rss < 0:
        raise ValueError(f"rss must be non-negative, got {rss}")
    aic = n * np.log(max(rss / n, 1e-300)) + 2 * k
    if n > k + 1:
        aic += 2 * k * (k + 1) / (n - k - 1)
    return float(aic)


def delta_aicc(
    lib_sizes: np.ndarray,
    score_mean: np.ndarray,
    score_var: np.ndarray,
    *,
    eps_var: float = 1e-12,
) -> float | None:
    """Compute delta-AICc between linear and saturation models.

    Parameters
    ----------
    lib_sizes : np.ndarray
        Library sizes tested.
    score_mean : np.ndarray
        Mean score at each library size.
    score_var : np.ndarray
        Variance of scores at each library size.
    eps_var : float
        Minimum variance floor for weights. Default is 1e-12.

    Returns
    -------
    float | None
        AICc(linear) - AICc(saturation), or None if fitting fails.
        Positive values favor the saturation model.
    """
    weights = 1.0 / np.maximum(score_var, eps_var)

    try:
        _, _, rss_linear = fit_linear(lib_sizes, score_mean, weights)
        _, _, _, rss_sat = fit_saturation(lib_sizes, score_mean, weights)
    except (np.linalg.LinAlgError, RuntimeError, ValueError):
        return None

    n = len(lib_sizes)
    return aicc(rss_linear, n, k=2) - aicc(rss_sat, n, k=3)
