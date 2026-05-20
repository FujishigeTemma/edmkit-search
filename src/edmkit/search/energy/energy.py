from collections.abc import Callable, Iterable

import numpy as np
import numpy.typing as npt

from edmkit.search.state import States

type Energies = npt.NDArray[np.float64]
"""1D array of energy values of shape ``(N,)``, one per state. Lower is better — strategies minimize energy."""

type Contexts = npt.NDArray[np.float64]
"""2D array of shape ``(N, K)`` carrying per-state auxiliary information between steps. The width ``K`` is fixed by the energy at construction time (``0`` when no context is needed); the contents are opaque to the rest of the search loop."""

type Plan = Callable[
    [States, Contexts],
    Iterable[Callable[[], tuple[slice, Energies, Contexts]]],
]
"""A factory that, given ``(states, contexts)``, yields a sequence of jobs covering the batch.

Each job is a zero-argument callable that returns ``(slice, energies, contexts)`` for the
contiguous slice it owns. Splitting work into independent jobs lets the caller execute them
however they like — sequentially, on a thread pool, on a process pool — without the plan
itself needing to know.
"""

type Energy = Callable[[States, Contexts], tuple[Energies, Contexts]]
"""A fully-applied energy: given a batch of states and their incoming contexts, return the new energies and contexts. Construct one by executing a `Plan` (e.g. via a thread pool)."""
