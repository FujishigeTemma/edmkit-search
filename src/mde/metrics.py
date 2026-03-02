import numpy as np

from .types import MetricFn


def validate(predictions: np.ndarray, observations: np.ndarray) -> None:
    """Validate inputs common to all metric functions."""
    if predictions.shape != observations.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs observations {observations.shape}"
        )
    if predictions.ndim != 2:
        raise ValueError(f"Expected 2D arrays (N, M), got {predictions.ndim}D")


def negate(metric: MetricFn) -> MetricFn:
    """Negate a metric function for use with lower-is-better metrics.

    Parameters
    ----------
    metric : MetricFn
        A metric function returning a scalar score.

    Returns
    -------
    MetricFn
        A new metric function that returns the negated score.
    """

    def negated(predictions: np.ndarray, observations: np.ndarray) -> float:
        return -metric(predictions, observations)

    negated.__name__ = f"negate({metric.__name__})"
    return negated


def mean_rho(predictions: np.ndarray, observations: np.ndarray) -> float:
    """Compute mean Pearson correlation across targets.

    Parameters
    ----------
    predictions : np.ndarray of shape (N, M)
        Predicted values where N is number of samples and M is number of targets.
    observations : np.ndarray of shape (N, M)
        Observed (ground truth) values.

    Returns
    -------
    float
        Mean correlation across all targets.
    """
    return float(np.mean(mean_rho_per_dim(predictions, observations)))


def rmse(predictions: np.ndarray, observations: np.ndarray) -> float:
    """Compute Root Mean Squared Error over all elements.

    Parameters
    ----------
    predictions : np.ndarray of shape (N, M)
        Predicted values where N is number of samples and M is number of targets.
    observations : np.ndarray of shape (N, M)
        Observed (ground truth) values.

    Returns
    -------
    float
        RMSE across all elements.
    """
    validate(predictions, observations)
    return float(np.sqrt(np.mean((predictions - observations) ** 2)))


def mae(predictions: np.ndarray, observations: np.ndarray) -> float:
    """Compute Mean Absolute Error over all elements.

    Parameters
    ----------
    predictions : np.ndarray of shape (N, M)
        Predicted values where N is number of samples and M is number of targets.
    observations : np.ndarray of shape (N, M)
        Observed (ground truth) values.

    Returns
    -------
    float
        MAE across all elements.
    """
    validate(predictions, observations)
    return float(np.mean(np.abs(predictions - observations)))


SCALAR_METRICS: tuple[MetricFn, ...] = (mean_rho, rmse, mae)


def mean_rho_per_dim(predictions: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation coefficient per target.

    Parameters
    ----------
    predictions : np.ndarray of shape (N, M)
        Predicted values where N is number of samples and M is number of targets.
    observations : np.ndarray of shape (N, M)
        Observed (ground truth) values.

    Returns
    -------
    per_target : np.ndarray of shape (M,)
        Correlation for each target.
    """
    validate(predictions, observations)
    predictions_c = predictions - predictions.mean(axis=0)
    observations_c = observations - observations.mean(axis=0)
    cov = (predictions_c * observations_c).sum(axis=0)
    denom = np.sqrt((predictions_c**2).sum(axis=0) * (observations_c**2).sum(axis=0))
    return np.where(denom > 0, cov / denom, 0.0)


def rmse_per_dim(predictions: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """Compute Root Mean Squared Error per target.

    Parameters
    ----------
    predictions : np.ndarray of shape (N, M)
        Predicted values where N is number of samples and M is number of targets.
    observations : np.ndarray of shape (N, M)
        Observed (ground truth) values.

    Returns
    -------
    per_target : np.ndarray of shape (M,)
        RMSE for each target.
    """
    validate(predictions, observations)
    return np.sqrt(np.mean((predictions - observations) ** 2, axis=0))


def mae_per_dim(predictions: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """Compute Mean Absolute Error per target.

    Parameters
    ----------
    predictions : np.ndarray of shape (N, M)
        Predicted values where N is number of samples and M is number of targets.
    observations : np.ndarray of shape (N, M)
        Observed (ground truth) values.

    Returns
    -------
    per_target : np.ndarray of shape (M,)
        MAE for each target.
    """
    validate(predictions, observations)
    return np.mean(np.abs(predictions - observations), axis=0)


PER_DIM_METRICS = (mean_rho_per_dim, rmse_per_dim, mae_per_dim)
