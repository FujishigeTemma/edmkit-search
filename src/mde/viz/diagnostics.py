"""Data diagnostic plotting functions."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_targets_timeseries(
    targets: np.ndarray,
    *,
    target_names: list[str],
) -> Figure:
    """Plot target time series with 2D scatter.

    Parameters
    ----------
    targets : np.ndarray of shape (T, 2)
        Target variable values (exactly 2 targets expected).
    target_names : list[str]
        Display names for the two targets.

    Returns
    -------
    Figure

    Raises
    ------
    ValueError
        If targets does not have shape (T, 2) or target_names has wrong length.
    """
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError(f"targets must have shape (T, 2), got {targets.shape}")
    if len(target_names) != 2:
        raise ValueError(
            f"target_names must have exactly 2 entries, got {len(target_names)}"
        )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    time = np.arange(targets.shape[0])
    col0 = targets[:, 0]
    col1 = targets[:, 1]

    axes[0].plot(time, col0, color="steelblue", linewidth=0.5)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel(target_names[0])
    axes[0].set_title(target_names[0])
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, col1, color="darkorange", linewidth=0.5)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel(target_names[1])
    axes[1].set_title(target_names[1])
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(col0, col1, alpha=0.3, s=1, c="purple")
    axes[2].set_xlabel(target_names[0])
    axes[2].set_ylabel(target_names[1])
    axes[2].set_title(f"{target_names[0]} vs {target_names[1]}")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_ts_overlay(
    data: np.ndarray,
    *,
    column_names: list[str],
) -> Figure:
    """Plot all time series variables overlaid on a single axes.

    Parameters
    ----------
    data : np.ndarray of shape (T, N)
        Time series data where N is the number of variables.
    column_names : list[str]
        Display names for each variable.

    Returns
    -------
    Figure
    """
    time = np.arange(data.shape[0])

    fig, ax = plt.subplots(figsize=(14, 6))

    for i in range(data.shape[1]):
        ax.plot(time, data[:, i], alpha=0.3, linewidth=0.3)

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(f"All TS Variables (n={len(column_names)})")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_distribution_stats(
    data: np.ndarray,
    *,
    column_names: list[str],
) -> Figure:
    """Plot histograms of per-variable mean and variance.

    Parameters
    ----------
    data : np.ndarray of shape (T, N)
        Time series data.
    column_names : list[str]
        Display names for each variable.

    Returns
    -------
    Figure
    """
    means = np.mean(data, axis=0)
    variances = np.var(data, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(means, bins=20, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Mean")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Variable Means")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(variances, bins=20, color="darkorange", edgecolor="white")
    axes[1].set_xlabel("Variance")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of Variable Variances")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_autocorrelation(
    data: np.ndarray,
    *,
    column_names: list[str],
    target_data: np.ndarray,
    target_names: list[str],
    lags: int = 200,
) -> Figure:
    """Plot autocorrelation functions for targets and sample variables.

    Parameters
    ----------
    data : np.ndarray of shape (T, N)
        Candidate time series data.
    column_names : list[str]
        Names for candidate variables.
    target_data : np.ndarray of shape (T, M)
        Target time series data.
    target_names : list[str]
        Names for target variables.
    lags : int, optional
        Number of lags to compute. Default is 200.

    Returns
    -------
    Figure
    """
    sample_indices = [0, data.shape[1] // 2, data.shape[1] - 1]
    sample_names = [column_names[i] for i in sample_indices]

    plot_names = list(target_names) + sample_names
    plot_series = [target_data[:, m] for m in range(target_data.shape[1])]
    plot_series += [data[:, i] for i in sample_indices]

    n = len(plot_names)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)

    for i, (name, series) in enumerate(zip(plot_names, plot_series)):
        x = series - series.mean()
        norm = np.dot(x, x)
        if norm == 0:
            acf = np.zeros(lags + 1)
        else:
            full_acf = np.correlate(x, x, mode="full")
            mid = len(full_acf) // 2
            acf = full_acf[mid : mid + lags + 1] / norm

        axes[i].plot(np.arange(lags + 1), acf, linewidth=0.8)
        axes[i].set_ylabel(name)
        axes[i].set_title(f"ACF: {name}")
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color="k", linewidth=0.5)

    axes[-1].set_xlabel("Lag")

    fig.tight_layout()
    return fig


def plot_correlation_heatmap(
    candidates: np.ndarray,
    targets: np.ndarray,
    *,
    candidate_names: list[str],
    target_names: list[str],
) -> Figure:
    """Plot Pearson correlation heatmap: candidates vs targets.

    Parameters
    ----------
    candidates : np.ndarray of shape (T, N)
        Candidate variable data.
    targets : np.ndarray of shape (T, M)
        Target variable data.
    candidate_names : list[str]
        Names for candidate variables.
    target_names : list[str]
        Names for target variables.

    Returns
    -------
    Figure
    """
    n_cands = candidates.shape[1]
    n_targets = targets.shape[1]
    N = candidates.shape[0]  # number of samples

    # Vectorized Pearson correlation: (n_cands, n_targets)
    c_centered = candidates - candidates.mean(axis=0)
    t_centered = targets - targets.mean(axis=0)
    c_std = np.sqrt((c_centered**2).sum(axis=0))
    t_std = np.sqrt((t_centered**2).sum(axis=0))
    c_std = np.where(c_std == 0, 1.0, c_std)
    t_std = np.where(t_std == 0, 1.0, t_std)
    corr = (c_centered / c_std).T @ (t_centered / t_std) / N

    fig, ax = plt.subplots(figsize=(6, 20))

    im = ax.imshow(corr, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="Correlation")

    ax.set_xticks(np.arange(n_targets))
    ax.set_yticks(np.arange(n_cands))
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(candidate_names, fontsize=6)

    ax.set_title("Pearson Correlation: Candidates vs Targets")
    ax.set_xlabel("Target")
    ax.set_ylabel("Candidate")

    fig.tight_layout()
    return fig
