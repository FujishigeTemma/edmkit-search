from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from edmkit.search.state import States

type Neighborhood = Callable[
    [States, np.random.Generator],
    tuple[States, npt.NDArray[np.int64]],
]
"""A function ``(parents, rng) -> (children, parents_idx)`` that expands a batch of ``N`` parent states into ``M`` children.

Given ``parents`` of shape ``(N, d)``, returns:

* ``children`` of shape ``(M, d')`` — the next-step states (typically ``d' = d + 1``).
* ``parents_idx`` of shape ``(M,)`` — ``parents_idx[i] in [0, N)`` points to the parent row
  that produced ``children[i]``.

The ``parents_idx`` array lets callers (e.g. `beam`) replicate parent-side data such as
the energy context in lockstep with the children, without the neighborhood having to know
about that data.
"""
