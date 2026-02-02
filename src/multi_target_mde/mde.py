"""MDE (Manifold Dimension Expansion) implementation.

MDE discovers causal relationships in multivariate time series data
and constructs optimal low-dimensional manifolds for prediction.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from edmkit import simplex_projection
from edmkit.embedding import lagged_embed

from .metrics import MetricFn, mean_rho
from .types import MDEResult
from .util import ensure_2d
from .validation import split_data


@dataclass
class _EvalResult:
    """Result of evaluating a single candidate variable."""

    var_idx: int
    score: float
    per_target_scores: np.ndarray


def _eval_candidate(args: tuple) -> _EvalResult | None:
    """Evaluate a single candidate variable (worker function for parallel execution)."""
    var_idx, selected_indices, X_train, X_val, Y_train, Y_val, threshold, metric = args

    candidate_indices = selected_indices + [var_idx]
    manifold_train = X_train[:, candidate_indices]
    manifold_val = X_val[:, candidate_indices]

    predictions = simplex_projection(manifold_train, Y_train, manifold_val)
    if predictions.ndim == 1:
        predictions = predictions[:, None]

    score, per_target_scores = metric(predictions, Y_val)

    if score < threshold:
        return None

    return _EvalResult(
        var_idx=var_idx, score=score, per_target_scores=per_target_scores
    )


def evaluate_manifold(
    manifold: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    query_indices: np.ndarray,
    *,
    metric: MetricFn = mean_rho,
) -> tuple[float, np.ndarray]:
    """Evaluate manifold prediction skill using simplex projection.

    Parameters
    ----------
    manifold : np.ndarray of shape (T, D)
        Current manifold where T is time points and D is dimensions.
    targets : np.ndarray of shape (T,) or (T, M)
        Target variables where M is number of targets.
    train_indices : np.ndarray
        Indices for training (library construction).
    query_indices : np.ndarray
        Indices for querying (skill evaluation).
    metric : MetricFn, optional
        Metric function to evaluate predictions. Default is mean_rho.

    Returns
    -------
    score : float
        Aggregate score across all targets.
    per_target_scores : np.ndarray of shape (M,)
        Score for each target dimension.
    """
    targets = ensure_2d(targets)

    X_lib = manifold[train_indices]
    Y_lib = targets[train_indices]  # (N, M) - all targets at once
    query_points = manifold[query_indices]

    # simplex_projection handles multi-target Y (N, M) -> predictions (Q, M)
    predictions = simplex_projection(X_lib, Y_lib, query_points)
    observations = targets[query_indices]

    return metric(predictions, observations)


def ccm_convergence_test(
    cause: np.ndarray,
    effect: np.ndarray,
    lib_sizes: list[int],
    *,
    E: int = 2,
    tau: int = 1,
    num_samples: int = 20,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """Test for CCM convergence to verify causal relationship.

    CCM principle: If X causes Y, then Y's manifold contains information
    about X. As library size increases, cross-mapping skill should converge.

    Supports multi-target: if cause/effect have multiple columns, computes
    mean convergence across all targets.

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
    rng : np.random.Generator | None, optional
        Random number generator.

    Returns
    -------
    rho_values : np.ndarray
        Array of mean rho at each library size.
    convergence_score : float
        rho[-1] - rho[0] (positive indicates convergence).
    """
    if rng is None:
        rng = np.random.default_rng()

    cause = ensure_2d(cause)
    effect = ensure_2d(effect)
    M = cause.shape[1]

    # Embed each effect column and align
    effect_embedded_list = []
    min_L = float("inf")

    for m in range(M):
        embedded = lagged_embed(effect[:, m], tau, E)
        effect_embedded_list.append(embedded)
        min_L = min(min_L, embedded.shape[0])

    min_L = int(min_L)
    offset = len(effect) - min_L

    # Align all embeddings and causes to min_L
    effect_embedded = np.stack(
        [e[-min_L:] for e in effect_embedded_list], axis=0
    )  # (M, L, E)
    cause_aligned = cause[offset:]  # (L, M)

    rho_values = np.zeros(len(lib_sizes))

    for i, lib_size in enumerate(lib_sizes):
        if lib_size >= min_L:
            lib_size = min_L - 1

        rho_samples = []
        for _ in range(num_samples):
            lib_indices = rng.choice(min_L, lib_size, replace=False)
            test_mask = np.ones(min_L, dtype=bool)
            test_mask[lib_indices] = False
            test_indices = np.where(test_mask)[0]

            if len(test_indices) == 0:
                continue

            # Compute rho for each target and average
            target_rhos = []
            for m in range(M):
                X_lib = effect_embedded[m, lib_indices]
                Y_lib = cause_aligned[lib_indices, m]
                query_points = effect_embedded[m, test_indices]
                observations = cause_aligned[test_indices, m]

                predictions = simplex_projection(X_lib, Y_lib, query_points)
                rho = np.corrcoef(observations, predictions)[0, 1]

                if not np.isnan(rho):
                    target_rhos.append(rho)

            if target_rhos:
                rho_samples.append(np.mean(target_rhos))

        rho_values[i] = np.mean(rho_samples) if rho_samples else 0.0

    convergence_score = rho_values[-1] - rho_values[0]
    return rho_values, convergence_score


def mde(
    candidates: np.ndarray,
    target: np.ndarray,
    *,
    metric: MetricFn = mean_rho,
    threshold: float = 0.3,
    max_dim: int = 10,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    gap: int = 0,
    ccm_validation: bool = False,
    ccm_threshold: float = 0.1,
    rng: np.random.Generator | None = None,
) -> MDEResult:
    """Perform Manifold Dimension Expansion to discover causal relationships.

    MDE greedily selects variables that maximize the given metric on the
    validation set. Supports both single-target and multi-target optimization.

    Parameters
    ----------
    candidates : np.ndarray of shape (T, N)
        Candidate variables where T is number of time points and N is number
        of candidate variables.
    target : np.ndarray of shape (T,) or (T, M)
        Target variable(s). Shape (T,) for single-target or (T, M) for
        multi-target where M is the number of target dimensions.
    metric : MetricFn, optional
        Metric function for evaluating prediction quality. Default is mean_rho.
        The function must return (aggregate_score, per_target_scores).
        MDE maximizes this score, so use negative values for metrics
        where lower is better (e.g., -RMSE).
    threshold : float, optional
        Minimum score for candidate selection. Default is 0.3.
    max_dim : int, optional
        Maximum manifold dimension. Default is 10.
    train_ratio : float, optional
        Ratio of data for training. Default is 0.6.
    val_ratio : float, optional
        Ratio of data for validation. Default is 0.2.
    gap : int, optional
        Number of time points to skip between splits. Default is 0.
    ccm_validation : bool, optional
        Whether to perform CCM validation. Default is False.
    ccm_threshold : float, optional
        Minimum CCM convergence threshold. Default is 0.1.
    rng : np.random.Generator | None, optional
        Random number generator for reproducibility.

    Returns
    -------
    MDEResult
        Result containing selected_indices, manifold, val_rhos, test_rhos,
        val_rhos_per_target, test_rhos_per_target, and ccm_scores.

    Raises
    ------
    ValueError
        If candidates is not 2D, max_dim exceeds N, or shapes mismatch.

    Examples
    --------
    >>> from multi_target_mde import mde, mean_rho
    >>> result = mde(candidates, target, metric=mean_rho, threshold=0.3)

    Using custom metric (minimize RMSE):

    >>> def neg_rmse(pred, obs):
    ...     rmse_val = np.sqrt(np.mean((pred - obs) ** 2, axis=0))
    ...     return -float(np.mean(rmse_val)), -rmse_val
    >>> result = mde(candidates, target, metric=neg_rmse, threshold=-0.5)
    """
    if candidates.ndim != 2:
        raise ValueError(
            f"candidates must be 2D array, got {candidates.ndim}D "
            f"with shape {candidates.shape}"
        )
    T, N = candidates.shape
    if max_dim > N:
        raise ValueError(f"max_dim must be <= N (={N}), got {max_dim}")

    target = ensure_2d(target)
    if target.shape[0] != T:
        raise ValueError(
            f"target length ({target.shape[0]}) must match candidates ({T})"
        )

    if rng is None:
        rng = np.random.default_rng()

    split = split_data(T, train_ratio, val_ratio, gap=gap)

    # Pre-slice data by split indices (avoids repeated row slicing in loops)
    X_train = candidates[split.train_indices]  # (N_train, N_candidates)
    X_val = candidates[split.val_indices]  # (N_val, N_candidates)
    X_test = candidates[split.test_indices]  # (N_test, N_candidates)
    Y_train = target[split.train_indices]  # (N_train, M)
    Y_val = target[split.val_indices]  # (N_val, M)
    Y_test = target[split.test_indices]  # (N_test, M)

    available_indices = list(range(N))
    selected_indices: list[int] = []
    val_scores: list[float] = []
    test_scores: list[float] = []
    val_scores_per_target: list[np.ndarray] = []
    test_scores_per_target: list[np.ndarray] = []
    ccm_scores: list[float] | None = [] if ccm_validation else None

    max_workers = os.cpu_count() or 1

    for _ in range(max_dim):
        best_candidate = None
        best_score = -np.inf
        best_per_target_scores: np.ndarray | None = None
        best_ccm_score: float | None = None

        eval_args = [
            (var_idx, selected_indices, X_train, X_val, Y_train, Y_val, threshold, metric)
            for var_idx in available_indices
        ]

        if max_workers > 1 and len(available_indices) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(_eval_candidate, eval_args))
        else:
            results = [_eval_candidate(args) for args in eval_args]

        for result in results:
            if result is None:
                continue

            score = result.score
            var_idx = result.var_idx
            per_target_scores = result.per_target_scores

            if ccm_validation:
                # Only run CCM test if this candidate could be the new best
                if score > best_score:
                    train_size = len(split.train_indices)
                    lib_sizes = [
                        int(train_size * r) for r in [0.1, 0.3, 0.5, 0.7, 0.9]
                    ]
                    # Ensure minimum library size and remove duplicates
                    lib_sizes = sorted(set(max(2, s) for s in lib_sizes))

                    cause = X_train[:, var_idx : var_idx + 1]

                    _, convergence_score = ccm_convergence_test(
                        cause=cause,
                        effect=Y_train,
                        lib_sizes=lib_sizes,
                        E=2,
                        tau=1,
                        num_samples=5,
                        rng=rng,
                    )

                    if convergence_score >= ccm_threshold:
                        best_candidate = var_idx
                        best_score = score
                        best_per_target_scores = per_target_scores
                        best_ccm_score = convergence_score
            else:
                if score > best_score:
                    best_candidate = var_idx
                    best_score = score
                    best_per_target_scores = per_target_scores

        if best_candidate is None:
            break

        selected_indices.append(best_candidate)
        available_indices.remove(best_candidate)
        val_scores.append(best_score)
        if best_per_target_scores is None:
            raise RuntimeError("best_per_target_scores should not be None here")
        val_scores_per_target.append(best_per_target_scores)

        if ccm_scores is not None and best_ccm_score is not None:
            ccm_scores.append(best_ccm_score)

        # Evaluate on test set using pre-sliced arrays
        manifold_train = X_train[:, selected_indices]
        manifold_test = X_test[:, selected_indices]
        test_pred = simplex_projection(manifold_train, Y_train, manifold_test)
        test_pred = ensure_2d(test_pred)
        test_score, test_per_target = metric(test_pred, Y_test)
        test_scores.append(test_score)
        test_scores_per_target.append(test_per_target)

    if len(selected_indices) == 0:
        return MDEResult(
            selected_indices=[],
            manifold=np.array([]).reshape(T, 0),
            val_rhos=[],
            test_rhos=[],
            val_rhos_per_target=[],
            test_rhos_per_target=[],
            ccm_scores=ccm_scores,
        )

    final_manifold = candidates[:, selected_indices]

    return MDEResult(
        selected_indices=selected_indices,
        manifold=final_manifold,
        val_rhos=val_scores,
        test_rhos=test_scores,
        val_rhos_per_target=val_scores_per_target,
        test_rhos_per_target=test_scores_per_target,
        ccm_scores=ccm_scores,
    )
