"""MDE results plotting functions."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from edmkit import simplex_projection

from ..validation import split_data


def plot_mde_results(
    mde_data: dict,
    output_path: Path,
    *,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    gap: int = 0,
) -> None:
    """Create line chart showing val vs test score at each dimension.

    Parameters
    ----------
    mde_data : dict
        Keys: target, val_rhos, test_rhos, selected_vars.
    output_path : Path
        Path to save the figure.
    """
    target = mde_data["target"]
    val_rhos = mde_data["val_rhos"]
    test_rhos = mde_data["test_rhos"]
    selected_vars = mde_data["selected_vars"]

    if len(val_rhos) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(1, len(val_rhos) + 1)

    ax.plot(x, val_rhos, "o-", label="Validation", color="steelblue", linewidth=2, markersize=8)
    ax.plot(x, test_rhos, "s--", label="Test", color="darkorange", linewidth=2, markersize=8)

    for i, (v, t) in enumerate(zip(val_rhos, test_rhos)):
        ax.annotate(f"{v:.3f}", xy=(x[i], v), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=9, color="steelblue")
        ax.annotate(f"{t:.3f}", xy=(x[i], t), xytext=(0, -15), textcoords="offset points",
                    ha="center", fontsize=9, color="darkorange")

    ax.set_xlabel("Dimension")
    ax.set_ylabel("Score")
    ax.set_title(f"MDE: {target} (Validation vs Test)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{i}\n({var})" for i, var in enumerate(selected_vars, 1)], fontsize=9)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    y_min = min(min(val_rhos), min(test_rhos))
    y_max = max(max(val_rhos), max(test_rhos))
    margin = (y_max - y_min) * 0.15
    ax.set_ylim(y_min - margin, y_max + margin)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_mde_results_multi(
    mde_data: dict,
    output_path: Path,
    *,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    gap: int = 0,
) -> None:
    """Create 3-panel line chart for multi-target MDE results.

    Panels: mean score, Left_Right score, FWD score.
    """
    val_rhos = mde_data["val_rhos"]
    test_rhos = mde_data["test_rhos"]
    val_rhos_per_target = mde_data["val_rhos_per_target"]
    test_rhos_per_target = mde_data["test_rhos_per_target"]
    selected_vars = mde_data["selected_vars"]

    if len(val_rhos) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(1, len(val_rhos) + 1)
    x_labels = [f"D{i}\n({var})" for i, var in enumerate(selected_vars, 1)]

    panels = [
        ("Mean Score", val_rhos, test_rhos),
        ("Left_Right", [r[0] for r in val_rhos_per_target], [r[0] for r in test_rhos_per_target]),
        ("FWD", [r[1] for r in val_rhos_per_target], [r[1] for r in test_rhos_per_target]),
    ]

    for ax, (title, val, test) in zip(axes, panels):
        ax.plot(x, val, "o-", label="Validation", color="steelblue", linewidth=2, markersize=8)
        ax.plot(x, test, "s--", label="Test", color="darkorange", linewidth=2, markersize=8)
        for i, (v, t) in enumerate(zip(val, test)):
            ax.annotate(f"{v:.3f}", xy=(x[i], v), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8, color="steelblue")
            ax.annotate(f"{t:.3f}", xy=(x[i], t), xytext=(0, -12), textcoords="offset points",
                        ha="center", fontsize=8, color="darkorange")
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"MDE: {mde_data['target']} (Validation vs Test)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_predictions(
    mde_data: dict,
    output_path: Path,
    *,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    gap: int = 0,
    split_name: str = "test",
) -> None:
    """Plot ground truth vs predictions at each dimension step.

    Parameters
    ----------
    mde_data : dict
        Keys: target, selected_indices, candidates, target_values, selected_vars.
    output_path : Path
        Path to save the figure.
    split_name : str
        Which split to plot predictions for: "val" or "test".
    """
    target_name = mde_data["target"]
    selected_indices = mde_data["selected_indices"]
    candidates = mde_data["candidates"]
    target_values = mde_data["target_values"]
    selected_vars = mde_data["selected_vars"]

    if len(selected_indices) == 0:
        return

    T = candidates.shape[0]
    split = split_data(T, train_ratio, val_ratio, gap=gap)

    query_indices = split.val_indices if split_name == "val" else split.test_indices
    ground_truth = target_values[query_indices]

    predictions_list = []
    for dim in range(1, len(selected_indices) + 1):
        manifold = candidates[:, selected_indices[:dim]]
        X_lib = manifold[split.train_indices]
        Y_lib = target_values[split.train_indices]
        preds = simplex_projection(X_lib, Y_lib, manifold[query_indices])
        predictions_list.append(preds)

    n_dims = len(predictions_list)
    fig, axes = plt.subplots(n_dims, 1, figsize=(14, 3 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]

    colors = matplotlib.colormaps["viridis"](np.linspace(0, 0.8, n_dims))

    for i, (pred, ax) in enumerate(zip(predictions_list, axes)):
        dim = i + 1
        vars_used = ", ".join(selected_vars[:dim])

        ax.plot(query_indices, ground_truth, "k-", linewidth=0.8, alpha=0.7, label="Ground Truth")
        ax.plot(query_indices, pred, "-", color=colors[i], linewidth=0.8, alpha=0.9,
                label=f"Prediction (D={dim})")

        valid_mask = ~np.isnan(pred)
        rho = np.corrcoef(ground_truth[valid_mask], pred[valid_mask])[0, 1] if valid_mask.sum() > 0 else np.nan

        ax.set_ylabel(target_name, fontsize=10)
        ax.set_title(f"D={dim}: {vars_used} (rho={rho:.4f})", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")
    plt.suptitle(f"{split_name.capitalize()} Predictions: {target_name}", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_predictions_multi(
    mde_data: dict,
    output_path: Path,
    *,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    gap: int = 0,
    split_name: str = "test",
) -> None:
    """Plot multi-target predictions at each dimension step.

    Grid layout: rows = dimensions, cols = targets.

    Parameters
    ----------
    split_name : str
        Which split to plot: "val" or "test".
    """
    selected_indices = mde_data["selected_indices"]
    candidates = mde_data["candidates"]
    targets = mde_data["target_values"]
    selected_vars = mde_data["selected_vars"]

    if len(selected_indices) == 0:
        return

    T = candidates.shape[0]
    split = split_data(T, train_ratio, val_ratio, gap=gap)
    query_indices = split.val_indices if split_name == "val" else split.test_indices

    M = targets.shape[1]
    target_names = ["Left_Right", "FWD"]
    observations = targets[query_indices]

    n_dims = len(selected_indices)
    fig, axes = plt.subplots(n_dims, M, figsize=(14, 3 * n_dims), sharex=True)
    if n_dims == 1:
        axes = axes.reshape(1, -1)

    colors = [matplotlib.colormaps["Blues"](0.7), matplotlib.colormaps["Oranges"](0.7)]

    for dim in range(1, n_dims + 1):
        manifold = candidates[:, selected_indices[:dim]]
        X_lib = manifold[split.train_indices]
        vars_used = ", ".join(selected_vars[:dim])

        for m in range(M):
            ax = axes[dim - 1, m]
            Y_lib = targets[split.train_indices, m]
            preds = simplex_projection(X_lib, Y_lib, manifold[query_indices])

            ax.plot(query_indices, observations[:, m], "k-", linewidth=0.8, alpha=0.7, label="Ground Truth")
            ax.plot(query_indices, preds, "-", color=colors[m], linewidth=0.8, alpha=0.9,
                    label=f"Prediction (D={dim})")

            valid_mask = ~np.isnan(preds)
            rho = np.corrcoef(observations[valid_mask, m], preds[valid_mask])[0, 1] if valid_mask.sum() > 0 else np.nan

            ax.set_title(f"D={dim}: {target_names[m]} (rho={rho:.4f})", fontsize=10)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

            if m == 0:
                ax.set_ylabel(f"D{dim}\n{vars_used}", fontsize=9)

    for m in range(M):
        axes[-1, m].set_xlabel("Time")

    plt.suptitle(f"Multi-Target {split_name.capitalize()} Predictions", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
