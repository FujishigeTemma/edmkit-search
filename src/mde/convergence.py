from collections.abc import Iterator

import numpy as np
from edmkit.ccm import ccm
from edmkit.embedding import lagged_embed

from .aicc import delta_aicc
from .types import PredictFn


def causation_iter(
    X: np.ndarray,
    Y: np.ndarray,
    lib_sizes: list[int],
    *,
    predict: PredictFn,
    E: int,
    tau: int,
    rng: np.random.Generator | None = None,
) -> Iterator[np.ndarray]:
    """Yield CCM score samples one at a time.

    Each yielded array has shape ``(len(lib_sizes),)`` containing the
    correlation at each library size for one bootstrap sample.
    The caller can consume as many samples as needed and stop early.

    Parameters
    ----------
    X : np.ndarray of shape (T,) or (T, M)
        Potential cause variable(s).
    Y : np.ndarray of shape (T,) or (T, M)
        Potential effect variable(s).
    lib_sizes : list[int]
        Library sizes to test (e.g. [10, 20, 50, 100]).
    predict : PredictFn
        Prediction function ``(X_train, Y_train, X_query) -> predictions``.
    E : int
        Embedding dimension.
    tau : int
        Time delay for embedding.
    rng : np.random.Generator or None
        Random number generator.

    Yields
    ------
    np.ndarray of shape (len(lib_sizes),)
        CCM correlation scores at each library size for one bootstrap sample.

    Raises
    ------
    ValueError
        If `E` or `tau` are not positive.
    """
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

    # To test X → Y: embed Y (effect), predict X (cause) from Y's attractor
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

    # TODO: edmkit の ccm がサンプル単位の情報を返せるようになったら、
    # 1回の呼び出しで全サンプルを取得する形に書き直す
    while True:
        # TODO: edmkit の ccm が metric を受け取れるようになったら、
        # causation() に metric パラメータを追加し、edmkit に渡す
        scores = ccm(
            Y_embedded,
            X_aligned,
            np.array(lib_sizes, dtype=int),
            predict_func=predict,
            n_samples=1,
            library_pool=library_pool,
            prediction_pool=prediction_pool,
            sampler=sampler,
        )
        yield scores


def causation(
    X: np.ndarray,
    Y: np.ndarray,
    lib_sizes: list[int],
    *,
    predict: PredictFn,
    E: int,
    tau: int,
    n_samples: int = 20,
    aicc_threshold: float = 4.0,
    rng: np.random.Generator | None = None,
) -> bool:
    """Test for CCM convergence using AICc model comparison.

    Collects ``n_samples`` bootstrap samples via :func:`causation_iter`,
    then compares a saturation model against a linear model using AICc.
    Returns ``True`` if the saturation model is favored (delta_aicc >= threshold),
    indicating convergent cross-mapping and thus causal influence from X on Y.

    Parameters
    ----------
    X : np.ndarray of shape (T,) or (T, M)
        Potential cause variable(s).
    Y : np.ndarray of shape (T,) or (T, M)
        Potential effect variable(s).
    lib_sizes : list[int]
        Library sizes to test (e.g. [10, 20, 50, 100]).
    predict : PredictFn
        Prediction function ``(X_train, Y_train, X_query) -> predictions``.
    E : int
        Embedding dimension.
    tau : int
        Time delay for embedding.
    n_samples : int
        Number of bootstrap samples to collect. Default is 20.
    aicc_threshold : float
        Minimum delta_aicc to declare convergence. Default is 4.0.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    bool
        True if convergence is detected (saturation model is favored).

    Raises
    ------
    ValueError
        If `lib_sizes` is empty or contains non-positive values,
        or if `n_samples` is not positive.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if not lib_sizes:
        raise ValueError("lib_sizes cannot be empty")
    if any(s <= 0 for s in lib_sizes):
        raise ValueError(f"All lib_sizes must be positive, got {lib_sizes}")

    all_scores = np.zeros((n_samples, len(lib_sizes)))
    for i, sample in zip(
        range(n_samples),
        causation_iter(
            X,
            Y,
            lib_sizes,
            predict=predict,
            E=E,
            tau=tau,
            rng=rng,
        ),
    ):
        all_scores[i] = sample

    means = all_scores.mean(axis=0)
    variances = all_scores.var(axis=0, ddof=1)

    delta = delta_aicc(np.array(lib_sizes, dtype=float), means, variances)
    if delta is None:
        return False

    return delta >= aicc_threshold
