import numpy as np
from edmkit.metrics import MetricFunc
from edmkit.splits import Fold
from edmkit.types import PredictFunc

from .dataset import Dataset, Subset


def negate(metric: MetricFunc) -> MetricFunc:
    """Negate a metric function for use with lower-is-better metrics.

    Parameters
    ----------
    metric : MetricFunc
        A metric function returning a score.

    Returns
    -------
    MetricFunc
        A new metric function that returns the negated score.
    """

    def negated(predictions: np.ndarray, observations: np.ndarray) -> np.ndarray:
        return -metric(predictions, observations)

    negated.__name__ = f"negate({getattr(metric, '__name__', repr(metric))})"
    return negated


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
    predict: PredictFunc,
    metric: MetricFunc,
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
    predict : PredictFunc
        Prediction function.
    metric : MetricFunc
        Metric function.

    Returns
    -------
    float
        Score for the given subset.
    """
    predictions = predict(X_train[:, indices], Y_train, X_validation[:, indices])
    if predictions.ndim == 1:
        predictions = predictions[:, None]

    return float(metric(predictions, Y_validation))


def score_subset_per_fold(
    indices: list[int],
    *,
    folds: list[Fold],
    X: np.ndarray,
    Y: np.ndarray,
    predict: PredictFunc,
    metric: MetricFunc,
) -> np.ndarray:
    """Score a variable subset on each fold independently.

    Parameters
    ----------
    indices : list[int]
        Column indices to use from X.
    folds : list[Fold]
        Temporal folds, each with ``.train`` and ``.validation`` index arrays.
    X : np.ndarray of shape (T, M)
        Full feature array (2D, already validated by caller).
    Y : np.ndarray of shape (T, D)
        Full target array (2D, already promoted by caller).
    predict : PredictFunc
        Prediction function.
    metric : MetricFunc
        Metric function.

    Returns
    -------
    np.ndarray of shape (n_folds,)
        Score for each fold.
    """
    scores = np.empty(len(folds))
    for k, fold in enumerate(folds):
        scores[k] = score_subset(
            indices,
            X_train=X[fold.train],
            X_validation=X[fold.validation],
            Y_train=Y[fold.train],
            Y_validation=Y[fold.validation],
            predict=predict,
            metric=metric,
        )
    return scores
