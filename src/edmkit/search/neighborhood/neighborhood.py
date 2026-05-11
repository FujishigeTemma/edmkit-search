from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from edmkit.search.state import States

type Neighborhood = Callable[
    [States, np.random.Generator],
    tuple[States, npt.NDArray[np.int64]],
]
"""
A Neighborhood expands a batch of N parent states into M children.

Given parents of shape (N, d), it returns:
  * `children`: shape (M, d') — the next-step states
  * `parents`:  shape (M,)    — `parents[i] in [0, N)` points to the parent row that
                                  produced `children[i]`

`parents` lets callers (e.g. beam search) replicate parent-side data such as the
energy context in lockstep with the children, without the neighborhood needing to
know about that data.
"""
