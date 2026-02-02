"""Data diagnostic plotting functions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from ..data import get_fly_columns


def plot_targets_timeseries(df: pl.DataFrame, output_path: Path) -> None:
    """Plot LEFT_RIGHT and FWD time series with 2D scatter.

    3 panels: LEFT_RIGHT TS, FWD TS, LEFT_RIGHT vs FWD scatter.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    time = np.arange(df.height)
    lr = df["Left_Right"].to_numpy()
    fwd = df["FWD"].to_numpy()

    axes[0].plot(time, lr, color="steelblue", linewidth=0.5)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Left_Right")
    axes[0].set_title("Left_Right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, fwd, color="darkorange", linewidth=0.5)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("FWD")
    axes[1].set_title("FWD")
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(lr, fwd, alpha=0.3, s=1, c="purple")
    axes[2].set_xlabel("Left_Right")
    axes[2].set_ylabel("FWD")
    axes[2].set_title("Left_Right vs FWD")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ts_overlay(df: pl.DataFrame, output_path: Path) -> None:
    """Plot all TS variables overlaid on a single axes."""
    ts_cols, _ = get_fly_columns(df)
    time = np.arange(df.height)

    fig, ax = plt.subplots(figsize=(14, 6))

    for col in ts_cols:
        ax.plot(time, df[col].to_numpy(), alpha=0.3, linewidth=0.3)

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(f"All TS Variables (n={len(ts_cols)})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_distribution_stats(df: pl.DataFrame, output_path: Path) -> None:
    """Plot histograms of per-variable mean and variance.

    2 panels: distribution of means, distribution of variances.
    """
    ts_cols, _ = get_fly_columns(df)

    means = np.array([df[col].mean() for col in ts_cols])
    variances = np.array([df[col].var() for col in ts_cols])

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

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_autocorrelation(
    df: pl.DataFrame, output_path: Path, *, lags: int = 200
) -> None:
    """Plot autocorrelation functions for targets and sample TS variables."""
    ts_cols, target_cols = get_fly_columns(df)
    sample_ts = [ts_cols[0], ts_cols[len(ts_cols) // 2], ts_cols[-1]]
    plot_cols = target_cols + sample_ts

    n = len(plot_cols)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)

    for i, col in enumerate(plot_cols):
        x = df[col].to_numpy()
        x = x - x.mean()
        norm = np.dot(x, x)
        if norm == 0:
            acf = np.zeros(lags + 1)
        else:
            full_acf = np.correlate(x, x, mode="full")
            mid = len(full_acf) // 2
            acf = full_acf[mid : mid + lags + 1] / norm

        axes[i].plot(np.arange(lags + 1), acf, linewidth=0.8)
        axes[i].set_ylabel(col)
        axes[i].set_title(f"ACF: {col}")
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color="k", linewidth=0.5)

    axes[-1].set_xlabel("Lag")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(df: pl.DataFrame, output_path: Path) -> None:
    """Plot Pearson correlation heatmap: all TS variables vs targets."""
    ts_cols, target_cols = get_fly_columns(df)

    data = np.zeros((len(ts_cols), len(target_cols)))
    for i, ts in enumerate(ts_cols):
        ts_arr = df[ts].to_numpy()
        for j, target in enumerate(target_cols):
            target_arr = df[target].to_numpy()
            data[i, j] = np.corrcoef(ts_arr, target_arr)[0, 1]

    fig, ax = plt.subplots(figsize=(6, 20))

    im = ax.imshow(data, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="Correlation")

    ax.set_xticks(np.arange(len(target_cols)))
    ax.set_yticks(np.arange(len(ts_cols)))
    ax.set_xticklabels(target_cols)
    ax.set_yticklabels(ts_cols, fontsize=6)

    ax.set_title("Pearson Correlation: TS Variables vs Targets")
    ax.set_xlabel("Target")
    ax.set_ylabel("TS Variable")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
