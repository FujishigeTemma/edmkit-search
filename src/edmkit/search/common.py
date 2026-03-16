import numpy as np

from .dataset import Dataset, Subset
from .types import MetricFn, PredictFn


def prepare_data(
    train: Dataset | Subset,
    validation: Dataset | Subset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract arrays, promote 1D Y to (N,1), validate X is 2D.

    Parameters
    ----------
    train : Dataset | Subset
        Training data. Only ``.X`` and ``.Y`` are accessed.
    validation : Dataset | Subset
        Validation data.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (X_train, X_validation, Y_train, Y_validation).

    Raises
    ------
    ValueError
        If X is not 2D.
    """
    X_train = train.X
    X_validation = validation.X
    Y_train = train.Y
    Y_validation = validation.Y

    if X_train.ndim != 2:
        raise ValueError(
            f"X_train must be 2D array, got {X_train.ndim}D with shape {X_train.shape}"
        )
    if X_validation.ndim != 2:
        raise ValueError(
            f"X_validation must be 2D array, got {X_validation.ndim}D with shape {X_validation.shape}"
        )

    if Y_train.ndim == 1:
        Y_train = Y_train[:, None]
    if Y_validation.ndim == 1:
        Y_validation = Y_validation[:, None]

    return X_train, X_validation, Y_train, Y_validation


def score_subset(
    indices: list[int],
    *,
    X_train: np.ndarray,
    X_validation: np.ndarray,
    Y_train: np.ndarray,
    Y_validation: np.ndarray,
    predict: PredictFn,
    metric: MetricFn,
) -> float:
    """Score a variable subset: predict then metric.

    Parameters
    ----------
    indices : list[int]
        Column indices to use from X.
    X_train : np.ndarray of shape (N, M)
        Training features.
    X_validation : np.ndarray of shape (N', M)
        Validation features.
    Y_train : np.ndarray of shape (N, D)
        Training targets.
    Y_validation : np.ndarray of shape (N', D)
        Validation targets.
    predict : PredictFn
        Prediction function.
    metric : MetricFn
        Metric function.

    Returns
    -------
    float
        Score for the given subset.
    """
    predictions = predict(X_train[:, indices], Y_train, X_validation[:, indices])
    if predictions.ndim == 1:
        predictions = predictions[:, None]

    return metric(predictions, Y_validation)
