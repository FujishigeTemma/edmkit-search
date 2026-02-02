"""Utility functions for multi-target MDE."""

import numpy as np


def ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 2D.

    Converts a 1D array of shape (T,) to 2D array of shape (T, 1).

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (T,) or (T, M).

    Returns
    -------
    np.ndarray
        Array of shape (T, M) where M >= 1.
    """
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets.

    Parameters
    ----------
    set1 : set
        First set.
    set2 : set
        Second set.

    Returns
    -------
    float
        Jaccard similarity coefficient (intersection / union).
        Returns 1.0 if both sets are empty.
    """
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)
