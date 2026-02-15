"""MDE results plotting functions."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .types import PlotData


def plot_mde_results(mde_data: PlotData) -> Figure:
    """Create line chart showing val vs test score at each dimension.

    Parameters
    ----------
    mde_data : PlotData
        Pre-computed MDE results.

    Returns
    -------
    Figure
    """
    val_scores = mde_data.train_scores
    test_scores = mde_data.val_scores
    selected_vars = mde_data.selected_var_names

    fig, ax = plt.subplots(figsize=(10, 6))

    if len(val_scores) == 0:
        return fig

    x = np.arange(1, len(val_scores) + 1)

    ax.plot(
        x,
        val_scores,
        "o-",
        label="Train",
        color="steelblue",
        linewidth=2,
        markersize=8,
    )
    ax.plot(
        x,
        test_scores,
        "s--",
        label="Validation",
        color="darkorange",
        linewidth=2,
        markersize=8,
    )

    for i, (v, t) in enumerate(zip(val_scores, test_scores)):
        ax.annotate(
            f"{v:.3f}",
            xy=(x[i], v),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="steelblue",
        )
        ax.annotate(
            f"{t:.3f}",
            xy=(x[i], t),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="darkorange",
        )

    ax.set_xlabel("Dimension")
    ax.set_ylabel("Score")
    ax.set_title(f"MDE: {mde_data.target_label} (Train vs Validation)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"D{i}\n({var})" for i, var in enumerate(selected_vars, 1)], fontsize=9
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    y_min = min(min(val_scores), min(test_scores))
    y_max = max(max(val_scores), max(test_scores))
    margin = (y_max - y_min) * 0.15
    ax.set_ylim(y_min - margin, y_max + margin)

    fig.tight_layout()
    return fig


def plot_mde_results_multi(
    mde_data: PlotData,
    *,
    target_names: list[str] | None = None,
) -> Figure:
    """Create multi-panel line chart for multi-target MDE results.

    Parameters
    ----------
    mde_data : PlotData
        Pre-computed MDE results.
    target_names : list[str] | None
        Display names for each target. If None, uses Target_0, Target_1, etc.

    Returns
    -------
    Figure
    """
    train_scores = mde_data.train_scores
    val_scores = mde_data.val_scores
    train_scores_per_target = mde_data.train_scores_per_target
    val_scores_per_target = mde_data.val_scores_per_target
    selected_vars = mde_data.selected_var_names

    if len(train_scores) == 0:
        return plt.figure()

    # Determine M from whichever per-target list is available
    has_train_per_target = len(train_scores_per_target) > 0
    has_val_per_target = len(val_scores_per_target) > 0
    if has_val_per_target:
        M = len(val_scores_per_target[0])
    elif has_train_per_target:
        M = len(train_scores_per_target[0])
    else:
        M = 0

    if target_names is None:
        target_names = [f"Target_{m}" for m in range(M)]

    n_panels = 1 + M
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    x = np.arange(1, len(train_scores) + 1)
    x_labels = [f"D{i}\n({var})" for i, var in enumerate(selected_vars, 1)]

    panels: list[tuple[str, list, list]] = [("Mean Score", train_scores, val_scores)]
    for m, name in enumerate(target_names):
        panels.append(
            (
                name,
                [r[m] for r in train_scores_per_target] if has_train_per_target else [],
                [r[m] for r in val_scores_per_target] if has_val_per_target else [],
            )
        )

    for ax, (title, train, val) in zip(axes, panels):
        if train:
            ax.plot(
                x,
                train,
                "o-",
                label="Train",
                color="steelblue",
                linewidth=2,
                markersize=8,
            )
        if val:
            ax.plot(
                x, val, "s--", label="Validation", color="darkorange", linewidth=2, markersize=8
            )
        for i in range(len(x)):
            if train and i < len(train):
                ax.annotate(
                    f"{train[i]:.3f}",
                    xy=(x[i], train[i]),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color="steelblue",
                )
            if val and i < len(val):
                ax.annotate(
                    f"{val[i]:.3f}",
                    xy=(x[i], val[i]),
                    xytext=(0, -12),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color="darkorange",
                )
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"MDE: {mde_data.target_label} (Train vs Validation)", fontsize=14)
    fig.tight_layout()
    return fig


def plot_predictions(mde_data: PlotData) -> Figure:
    """Plot ground truth vs predictions at each dimension step.

    Parameters
    ----------
    mde_data : PlotData
        Pre-computed MDE results including query_indices, observations,
        and predictions_per_dim.

    Returns
    -------
    Figure
    """
    target_name = mde_data.target_label
    selected_vars = mde_data.selected_var_names
    query_indices = mde_data.query_indices
    ground_truth = mde_data.observations
    predictions_list = mde_data.predictions_per_dim

    if len(predictions_list) == 0:
        return plt.figure()

    # Squeeze single-target predictions to 1D
    predictions_list = [
        p.squeeze() if p.ndim == 2 and p.shape[1] == 1 else p for p in predictions_list
    ]
    if ground_truth.ndim == 2 and ground_truth.shape[1] == 1:
        ground_truth = ground_truth.squeeze()

    n_dims = len(predictions_list)
    fig, axes = plt.subplots(n_dims, 1, figsize=(14, 3 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]

    colors = matplotlib.colormaps["viridis"](np.linspace(0, 0.8, n_dims))

    for i, (prediction, ax) in enumerate(zip(predictions_list, axes)):
        dim = i + 1
        vars_used = ", ".join(selected_vars[:dim])

        ax.plot(
            query_indices,
            ground_truth,
            "k-",
            linewidth=0.8,
            alpha=0.7,
            label="Ground Truth",
        )
        ax.plot(
            query_indices,
            prediction,
            "-",
            color=colors[i],
            linewidth=0.8,
            alpha=0.9,
            label=f"Prediction (D={dim})",
        )

        valid_mask = ~np.isnan(prediction)
        rho = (
            np.corrcoef(ground_truth[valid_mask], prediction[valid_mask])[0, 1]
            if valid_mask.sum() > 1
            else np.nan
        )

        ax.set_ylabel(target_name, fontsize=10)
        ax.set_title(f"D={dim}: {vars_used} (rho={rho:.4f})", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")
    fig.suptitle(f"Predictions: {target_name}", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


def plot_predictions_multi(
    mde_data: PlotData,
    *,
    target_names: list[str] | None = None,
) -> Figure:
    """Plot multi-target predictions at each dimension step.

    Parameters
    ----------
    mde_data : PlotData
        Pre-computed MDE results.
    target_names : list[str] | None
        Display names for each target.

    Returns
    -------
    Figure
    """
    selected_vars = mde_data.selected_var_names
    query_indices = mde_data.query_indices
    observations = mde_data.observations
    predictions_list = mde_data.predictions_per_dim

    if len(predictions_list) == 0:
        return plt.figure()

    M = observations.shape[1]
    if target_names is None:
        target_names = [f"Target_{m}" for m in range(M)]

    n_dims = len(predictions_list)
    fig, axes = plt.subplots(n_dims, M, figsize=(7 * M, 3 * n_dims), sharex=True)
    if n_dims == 1:
        axes = axes.reshape(1, -1)
    if M == 1:
        axes = axes.reshape(-1, 1)

    cmap_names = ["Blues", "Oranges", "Greens", "Reds", "Purples"]
    colors = [
        matplotlib.colormaps[cmap_names[m % len(cmap_names)]](0.7) for m in range(M)
    ]

    for dim_idx in range(n_dims):
        dim = dim_idx + 1
        vars_used = ", ".join(selected_vars[:dim])
        predictions = predictions_list[dim_idx]

        for m in range(M):
            ax = axes[dim_idx, m]
            prediction_m = predictions[:, m] if predictions.ndim == 2 else predictions

            ax.plot(
                query_indices,
                observations[:, m],
                "k-",
                linewidth=0.8,
                alpha=0.7,
                label="Ground Truth",
            )
            ax.plot(
                query_indices,
                prediction_m,
                "-",
                color=colors[m],
                linewidth=0.8,
                alpha=0.9,
                label=f"Prediction (D={dim})",
            )

            valid_mask = ~np.isnan(prediction_m)
            rho = (
                np.corrcoef(observations[valid_mask, m], prediction_m[valid_mask])[0, 1]
                if valid_mask.sum() > 1
                else np.nan
            )

            ax.set_title(f"D={dim}: {target_names[m]} (rho={rho:.4f})", fontsize=10)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

            if m == 0:
                ax.set_ylabel(f"D{dim}\n{vars_used}", fontsize=9)

    for m in range(M):
        axes[-1, m].set_xlabel("Time")

    fig.suptitle("Multi-Target Predictions", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig
