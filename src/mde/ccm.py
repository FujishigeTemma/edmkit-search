"""CCM (Convergent Cross-Mapping) convergence test."""

import numpy as np
from edmkit.embedding import lagged_embed
from scipy.optimize import least_squares

from .metrics import MetricFn
from .skill import PredictFn, _ensure_2d


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prepare_ccm_embedding(
    effect: np.ndarray,
    cause: np.ndarray,
    tau: int,
    E: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare lagged embedding and align arrays.

    Parameters
    ----------
    effect : np.ndarray of shape (T, M)
        Effect variable(s) to embed.
    cause : np.ndarray of shape (T, M)
        Cause variable(s) to align.
    tau : int
        Time delay for embedding.
    E : int
        Embedding dimension.

    Returns
    -------
    effect_embedded : np.ndarray of shape (M, L, E)
        Embedded effect variables.
    cause_aligned : np.ndarray of shape (L, M)
        Aligned cause variables.
    """
    M = effect.shape[1]

    effect_embedded_list = []
    min_L = float("inf")

    for m in range(M):
        embedded = lagged_embed(effect[:, m], tau, E)
        effect_embedded_list.append(embedded)
        min_L = min(min_L, embedded.shape[0])

    min_L = int(min_L)
    offset = len(effect) - min_L

    effect_embedded = np.stack(
        [e[-min_L:] for e in effect_embedded_list], axis=0
    )  # (M, L, E)
    cause_aligned = cause[offset:]  # (L, M)

    return effect_embedded, cause_aligned


def _sample_ccm_scores(
    cause: np.ndarray,
    effect_embedded: np.ndarray,
    lib_sizes: list[int],
    num_samples: int,
    predict: PredictFn,
    metric: MetricFn,
    rng: np.random.Generator,
) -> list[list[float]]:
    """Sample CCM scores at each library size.

    Parameters
    ----------
    cause : np.ndarray of shape (L, M_cause)
        Aligned cause variable(s).
    effect_embedded : np.ndarray of shape (M_effect, L, E)
        Embedded effect variables.
    lib_sizes : list[int]
        Library sizes to test.
    num_samples : int
        Number of random samples per library size.
    predict : PredictFn
        Prediction function.
    metric : MetricFn
        Metric function.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    all_samples : list[list[float]]
        Scores at each library size.

    Notes
    -----
    When M_cause == 1 and M_effect > 1, the single cause variable is predicted
    from each effect manifold independently.
    """
    M_effect = effect_embedded.shape[0]
    M_cause = cause.shape[1]
    min_L = effect_embedded.shape[1]

    all_samples: list[list[float]] = [[] for _ in range(len(lib_sizes))]

    for i, lib_size in enumerate(lib_sizes):
        actual_lib_size = min(lib_size, min_L - 1)

        for _ in range(num_samples):
            train_indices = rng.choice(min_L, actual_lib_size, replace=False)
            test_mask = np.ones(min_L, dtype=bool)
            test_mask[train_indices] = False
            test_indices = np.where(test_mask)[0]

            if len(test_indices) == 0:
                continue

            all_predictions = []
            all_observations = []
            for m in range(M_effect):
                # Use corresponding cause column, or column 0 if cause is univariate
                cause_idx = m if M_cause > 1 else 0
                X_train = effect_embedded[m, train_indices]
                Y_train = cause[train_indices, cause_idx]
                X_test = effect_embedded[m, test_indices]
                observations = cause[test_indices, cause_idx]

                predictions = predict(X_train, Y_train, X_test)
                all_predictions.append(predictions)
                all_observations.append(observations)

            prediction_matrix = np.column_stack(all_predictions)
            observation_matrix = np.column_stack(all_observations)
            score = metric(prediction_matrix, observation_matrix)
            all_samples[i].append(score)

    return all_samples


def _compute_sample_statistics(
    samples: list[list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and variance from samples.

    Parameters
    ----------
    samples : list[list[float]]
        Scores at each library size.

    Returns
    -------
    means : np.ndarray
        Mean score at each library size.
    variances : np.ndarray
        Variance of scores at each library size.
    """
    n = len(samples)
    means = np.zeros(n)
    variances = np.zeros(n)

    for i in range(n):
        if samples[i]:
            means[i] = np.mean(samples[i])
            variances[i] = (
                np.var(samples[i], ddof=1) if len(samples[i]) > 1 else 0.0
            )
        else:
            means[i] = 0.0
            variances[i] = 0.0

    return means, variances


def _fit_linear(
    L: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float]:
    """Fit linear model s(L) = alpha + beta * L.

    Parameters
    ----------
    L : np.ndarray
        Library sizes.
    scores : np.ndarray
        Mean scores at each library size.
    weights : np.ndarray
        Weights for weighted least squares.

    Returns
    -------
    alpha : float
        Intercept.
    beta : float
        Slope.
    rss : float
        Residual sum of squares (unweighted).
    """
    A = np.column_stack([np.ones_like(L), L])
    Aw = A * weights[:, None]
    params = np.linalg.solve(Aw.T @ A, Aw.T @ scores)
    alpha, beta = params
    residuals = scores - (alpha + beta * L)
    rss = float(np.sum(residuals**2))
    return alpha, beta, rss


def _fit_saturation(
    L: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float, float]:
    """Fit saturation model s(L) = a - b * exp(-c * L).

    b and c are constrained positive via exp reparameterization.

    Parameters
    ----------
    L : np.ndarray
        Library sizes.
    scores : np.ndarray
        Mean scores at each library size.
    weights : np.ndarray
        Weights for weighted least squares.

    Returns
    -------
    a : float
        Asymptotic value.
    b : float
        Scale parameter.
    c : float
        Rate parameter.
    rss : float
        Residual sum of squares (unweighted).
    """
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


def _aicc(rss: float, n: int, k: int) -> float:
    """Compute AICc (or AIC fallback when n <= k + 1).

    Parameters
    ----------
    rss : float
        Residual sum of squares.
    n : int
        Number of observations.
    k : int
        Number of parameters.

    Returns
    -------
    float
        AICc value.
    """
    aic = n * np.log(max(rss / n, 1e-300)) + 2 * k
    if n > k + 1:
        aic += 2 * k * (k + 1) / (n - k - 1)
    return float(aic)


def _delta_aicc(
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
    eps_var : float, optional
        Minimum variance floor for weights. Default is 1e-12.

    Returns
    -------
    float | None
        AICc_linear - AICc_saturation, or None if fitting fails.
        Positive values favor the saturation model.
    """
    weights = 1.0 / np.maximum(score_var, eps_var)

    try:
        _, _, rss_linear = _fit_linear(lib_sizes, score_mean, weights)
        _, _, _, rss_sat = _fit_saturation(lib_sizes, score_mean, weights)
    except (np.linalg.LinAlgError, RuntimeError, ValueError):
        return None

    n = len(lib_sizes)
    return _aicc(rss_linear, n, k=2) - _aicc(rss_sat, n, k=3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def converged(
    cause: np.ndarray,
    effect: np.ndarray,
    lib_sizes: list[int],
    *,
    E: int = 2,
    tau: int = 1,
    num_samples: int = 20,
    predict: PredictFn,
    metric: MetricFn,
    aicc_threshold: float = 4.0,
    rng: np.random.Generator | None = None,
) -> bool:
    """Test for CCM convergence.

    Compares a saturation model s(L) = a - b*exp(-c*L) against a linear model
    s(L) = alpha + beta*L via AICc. Returns True if saturation is favored
    (delta_aicc >= threshold).

    Parameters
    ----------
    cause : np.ndarray of shape (T,) or (T, M)
        Potential cause variable(s).
    effect : np.ndarray of shape (T,) or (T, M)
        Potential effect variable(s).
    lib_sizes : list[int]
        Library sizes to test (e.g., [10, 20, 50, 100]).
    E : int, optional
        Embedding dimension. Default is 2.
    tau : int, optional
        Time delay. Default is 1.
    num_samples : int, optional
        Number of random samples per library size. Default is 20.
    predict : PredictFn
        Prediction function (X_train, Y_train, X_query) -> predictions.
    metric : MetricFn
        Metric function to evaluate predictions.
    aicc_threshold : float, optional
        Minimum delta_aicc to declare convergence. Default is 4.0.
    rng : np.random.Generator | None, optional
        Random number generator.

    Returns
    -------
    bool
        True if convergence is detected (saturation model is favored).

    Raises
    ------
    ValueError
        If lib_sizes is empty or contains non-positive values.
    """
    if not lib_sizes:
        raise ValueError("lib_sizes cannot be empty")
    if any(L <= 0 for L in lib_sizes):
        raise ValueError(f"All lib_sizes must be positive, got {lib_sizes}")

    if rng is None:
        rng = np.random.default_rng()

    cause_2d = _ensure_2d(cause)
    effect_2d = _ensure_2d(effect)
    effect_embedded, cause_aligned = _prepare_ccm_embedding(effect_2d, cause_2d, tau, E)

    samples = _sample_ccm_scores(
        cause_aligned, effect_embedded, lib_sizes, num_samples, predict, metric, rng
    )
    score_mean, score_var = _compute_sample_statistics(samples)

    delta = _delta_aicc(np.array(lib_sizes, dtype=float), score_mean, score_var)
    if delta is None:
        return False

    return delta >= aicc_threshold


def make_filter(
    target: np.ndarray,
    train_indices: np.ndarray,
    *,
    predict: PredictFn,
    metric: MetricFn,
    E: int = 2,
    tau: int = 1,
    num_samples: int = 20,
    aicc_threshold: float = 4.0,
    rng: np.random.Generator | None = None,
):
    """Create a candidate filter that checks CCM convergence.

    Returns a filter function compatible with ``greedy(candidate_filter=...)``.
    The filter tests whether adding a candidate variable shows convergent
    cross-mapping behavior.

    Parameters
    ----------
    target : np.ndarray of shape (T,) or (T, M)
        Target variable(s) that the candidate should predict.
    train_indices : np.ndarray
        Training indices used for CCM testing.
    predict : PredictFn
        Prediction function for CCM.
    metric : MetricFn
        Metric function for CCM evaluation.
    E : int, optional
        Embedding dimension for CCM. Default is 2.
    tau : int, optional
        Time delay for CCM embedding. Default is 1.
    num_samples : int, optional
        Number of samples per library size. Default is 20.
    aicc_threshold : float, optional
        Minimum delta_aicc to accept convergence. Default is 4.0.
    rng : np.random.Generator | None, optional
        Random number generator for reproducibility.

    Returns
    -------
    Filter
        A filter function with signature
        ``(var_idx, X_train, Y_train, selected_indices) -> bool``.

    Examples
    --------
    >>> from mde.ccm import make_filter
    >>> from mde import greedy, mean_rho
    >>> from edmkit import simplex_projection
    >>> ccm_filter = make_filter(
    ...     target, train_indices,
    ...     predict=simplex_projection, metric=mean_rho
    ... )
    >>> result = greedy(
    ...     candidates, target, train_indices, val_indices,
    ...     predict=simplex_projection, metric=mean_rho,
    ...     candidate_filter=ccm_filter,
    ... )
    """
    if rng is None:
        rng = np.random.default_rng()

    def ccm_filter(
        var_idx: int,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        selected_indices: list[int],
    ) -> bool:
        """Check if candidate shows CCM convergence."""
        cause = X_train[:, var_idx : var_idx + 1]
        n = len(X_train)

        lib_sizes = np.unique(
            np.geomspace(10, max(10, n * 0.8), num=8).astype(int)
        ).tolist()
        lib_sizes = [max(2, s) for s in lib_sizes]

        return converged(
            cause=cause,
            effect=Y_train,
            lib_sizes=lib_sizes,
            E=E,
            tau=tau,
            num_samples=num_samples,
            predict=predict,
            metric=metric,
            aicc_threshold=aicc_threshold,
            rng=rng,
        )

    return ccm_filter
