"""CCM (Convergent Cross-Mapping) convergence test."""

from typing import NamedTuple

import numpy as np
from edmkit.embedding import lagged_embed
from scipy.optimize import least_squares

from .metrics import MetricFn
from .skill import PredictFn, _ensure_2d


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


class _CCMSampleStats(NamedTuple):
    """Internal: sampling statistics."""

    lib_sizes: np.ndarray
    score_mean: np.ndarray
    score_var: np.ndarray


class _CCMModelFit(NamedTuple):
    """Internal: model fitting results."""

    aicc_saturation: float
    aicc_linear: float
    delta_aicc: float
    saturation_params: tuple[float, float, float]
    linear_params: tuple[float, float]


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


class CCMDiagnostics(NamedTuple):
    """Diagnostics for CCM convergence test.

    Parameters
    ----------
    lib_sizes : np.ndarray
        Shape (n_lib_sizes,) - library sizes tested.
    score_mean : np.ndarray
        Shape (n_lib_sizes,) - mean score at each library size.
    score_var : np.ndarray
        Shape (n_lib_sizes,) - variance of scores at each library size.
    aicc_saturation : float
        AICc of the saturation model.
    aicc_linear : float
        AICc of the linear model.
    delta_aicc : float
        AICc_linear - AICc_saturation. Positive means saturation is favored.
    saturation_params : tuple[float, float, float]
        Fitted (a, b, c) for s(L) = a - b * exp(-c * L).
    linear_params : tuple[float, float]
        Fitted (alpha, beta) for s(L) = alpha + beta * L.
    """

    lib_sizes: np.ndarray
    score_mean: np.ndarray
    score_var: np.ndarray
    aicc_saturation: float
    aicc_linear: float
    delta_aicc: float
    saturation_params: tuple[float, float, float]
    linear_params: tuple[float, float]


# ---------------------------------------------------------------------------
# Internal helper functions
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
            lib_indices = rng.choice(min_L, actual_lib_size, replace=False)
            test_mask = np.ones(min_L, dtype=bool)
            test_mask[lib_indices] = False
            test_indices = np.where(test_mask)[0]

            if len(test_indices) == 0:
                continue

            all_preds = []
            all_obs = []
            for m in range(M_effect):
                # Use corresponding cause column, or column 0 if cause is univariate
                cause_idx = m if M_cause > 1 else 0
                X_lib = effect_embedded[m, lib_indices]
                Y_lib = cause[lib_indices, cause_idx]
                query_points = effect_embedded[m, test_indices]
                observations = cause[test_indices, cause_idx]

                predictions = predict(X_lib, Y_lib, query_points)
                all_preds.append(predictions)
                all_obs.append(observations)

            pred_matrix = np.column_stack(all_preds)
            obs_matrix = np.column_stack(all_obs)
            score, _ = metric(pred_matrix, obs_matrix)
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
        pred = a - b * np.exp(-c * L)
        return sqrt_w * (scores - pred)

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


def _compute_ccm_statistics(
    cause: np.ndarray,
    effect: np.ndarray,
    lib_sizes: list[int],
    *,
    E: int,
    tau: int,
    num_samples: int,
    predict: PredictFn,
    metric: MetricFn,
    rng: np.random.Generator,
) -> _CCMSampleStats:
    """Compute CCM sampling statistics.

    Parameters
    ----------
    cause : np.ndarray of shape (T,) or (T, M)
        Potential cause variable(s).
    effect : np.ndarray of shape (T,) or (T, M)
        Potential effect variable(s).
    lib_sizes : list[int]
        Library sizes to test.
    E : int
        Embedding dimension.
    tau : int
        Time delay.
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
    _CCMSampleStats
        Sampling statistics (lib_sizes, score_mean, score_var).
    """
    cause_2d = _ensure_2d(cause)
    effect_2d = _ensure_2d(effect)

    effect_embedded, cause_aligned = _prepare_ccm_embedding(effect_2d, cause_2d, tau, E)

    all_samples = _sample_ccm_scores(
        cause_aligned, effect_embedded, lib_sizes, num_samples, predict, metric, rng
    )

    score_mean, score_var = _compute_sample_statistics(all_samples)

    return _CCMSampleStats(
        lib_sizes=np.array(lib_sizes, dtype=float),
        score_mean=score_mean,
        score_var=score_var,
    )


def _fit_convergence_models(
    stats: _CCMSampleStats,
    *,
    eps_var: float = 1e-12,
) -> _CCMModelFit | None:
    """Fit convergence models and compute AICc.

    Parameters
    ----------
    stats : _CCMSampleStats
        Sampling statistics.
    eps_var : float, optional
        Minimum variance floor for weights. Default is 1e-12.

    Returns
    -------
    _CCMModelFit | None
        Model fitting results, or None if fitting fails.
    """
    weights = 1.0 / np.maximum(stats.score_var, eps_var)

    try:
        alpha, beta, rss_linear = _fit_linear(
            stats.lib_sizes, stats.score_mean, weights
        )
        a, b, c, rss_sat = _fit_saturation(stats.lib_sizes, stats.score_mean, weights)
    except (np.linalg.LinAlgError, RuntimeError, ValueError):
        return None

    n = len(stats.lib_sizes)
    aicc_linear = _aicc(rss_linear, n, k=2)
    aicc_sat = _aicc(rss_sat, n, k=3)
    delta_aicc = aicc_linear - aicc_sat

    return _CCMModelFit(
        aicc_saturation=aicc_sat,
        aicc_linear=aicc_linear,
        delta_aicc=delta_aicc,
        saturation_params=(a, b, c),
        linear_params=(alpha, beta),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ccm_converged(
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
    """Test for CCM convergence (minimal interface).

    Compares a saturation model s(L) = a - b*exp(-c*L) against a linear model
    s(L) = alpha + beta*L via AICc. Returns True if saturation is favored
    (delta_aicc >= threshold).

    For diagnostic information (scores, AICc values, fitted parameters),
    use ``ccm_convergence_diagnostics()`` instead.

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
        Prediction function (X_lib, Y_lib, X_query) -> predictions.
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

    stats = _compute_ccm_statistics(
        cause,
        effect,
        lib_sizes,
        E=E,
        tau=tau,
        num_samples=num_samples,
        predict=predict,
        metric=metric,
        rng=rng,
    )

    fit = _fit_convergence_models(stats)
    if fit is None:
        return False

    return fit.delta_aicc >= aicc_threshold


def ccm_convergence_diagnostics(
    cause: np.ndarray,
    effect: np.ndarray,
    lib_sizes: list[int],
    *,
    E: int = 2,
    tau: int = 1,
    num_samples: int = 20,
    predict: PredictFn,
    metric: MetricFn,
    rng: np.random.Generator | None = None,
) -> CCMDiagnostics:
    """Compute CCM convergence diagnostics for visualization and analysis.

    Use this function when you need detailed information about the convergence
    test (e.g., for plotting convergence curves or analyzing model fits).
    For simple convergence testing, use ``ccm_converged()`` instead.

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
        Prediction function (X_lib, Y_lib, X_query) -> predictions.
    metric : MetricFn
        Metric function to evaluate predictions.
    rng : np.random.Generator | None, optional
        Random number generator.

    Returns
    -------
    CCMDiagnostics
        Diagnostic information including scores, AICc values, and fitted params.

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

    stats = _compute_ccm_statistics(
        cause,
        effect,
        lib_sizes,
        E=E,
        tau=tau,
        num_samples=num_samples,
        predict=predict,
        metric=metric,
        rng=rng,
    )

    fit = _fit_convergence_models(stats)
    if fit is None:
        return CCMDiagnostics(
            lib_sizes=stats.lib_sizes,
            score_mean=stats.score_mean,
            score_var=stats.score_var,
            aicc_saturation=np.inf,
            aicc_linear=np.inf,
            delta_aicc=0.0,
            saturation_params=(0.0, 0.0, 0.0),
            linear_params=(0.0, 0.0),
        )

    return CCMDiagnostics(
        lib_sizes=stats.lib_sizes,
        score_mean=stats.score_mean,
        score_var=stats.score_var,
        aicc_saturation=fit.aicc_saturation,
        aicc_linear=fit.aicc_linear,
        delta_aicc=fit.delta_aicc,
        saturation_params=fit.saturation_params,
        linear_params=fit.linear_params,
    )


def make_ccm_filter(
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

    Returns a filter function compatible with ``greedy_select(candidate_filter=...)``.
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
    CandidateFilter
        A filter function with signature
        ``(var_idx, X_train, Y_train, selected_indices) -> bool``.

    Examples
    --------
    >>> from mde import greedy_select, make_ccm_filter, mean_rho
    >>> from edmkit import simplex_projection
    >>> ccm_filter = make_ccm_filter(
    ...     target, train_indices,
    ...     predict=simplex_projection, metric=mean_rho
    ... )
    >>> selected, scores, _, _ = greedy_select(
    ...     candidates, target, train_indices, val_indices,
    ...     predict=simplex_projection, metric=mean_rho,
    ...     candidate_filter=ccm_filter,
    ... )
    """
    if rng is None:
        rng = np.random.default_rng()

    target_2d = _ensure_2d(target)
    Y_train_full = target_2d[train_indices]
    train_size = len(train_indices)

    # Log-spaced library sizes for saturation detection
    lib_sizes = np.unique(
        np.geomspace(10, max(10, train_size * 0.8), num=8).astype(int)
    ).tolist()
    lib_sizes = [max(2, s) for s in lib_sizes]

    def ccm_filter(
        var_idx: int,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        selected_indices: list[int],
    ) -> bool:
        """Check if candidate shows CCM convergence."""
        cause = X_train[:, var_idx : var_idx + 1]

        return ccm_converged(
            cause=cause,
            effect=Y_train_full,
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
