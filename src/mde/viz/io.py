"""I/O utilities for visualization."""

from pathlib import Path

from matplotlib.figure import Figure


def save_figure(fig: Figure, path: Path | str, *, dpi: int = 150) -> None:
    """Save a matplotlib Figure to disk and close it.

    Parameters
    ----------
    fig : Figure
        The figure to save.
    path : Path | str
        Output file path.
    dpi : int, optional
        Resolution in dots per inch. Default is 150.
    """
    import matplotlib.pyplot as plt

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
