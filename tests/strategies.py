from typing import TYPE_CHECKING

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

reasonable_floats = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)


if TYPE_CHECKING:

    def arrays_2d(*, min_n: int = ..., max_n: int = ..., min_m: int = ..., max_m: int = ...) -> st.SearchStrategy[np.ndarray]: ...

    def matched_arrays(
        *, min_n: int = ..., max_n: int = ..., min_m: int = ..., max_m: int = ...
    ) -> st.SearchStrategy[tuple[np.ndarray, np.ndarray]]: ...

else:

    @st.composite
    def arrays_2d(draw, *, min_n=2, max_n=100, min_m=1, max_m=10):
        """Generate 2D float64 arrays of shape (N, M)."""
        n = draw(st.integers(min_value=min_n, max_value=max_n))
        m = draw(st.integers(min_value=min_m, max_value=max_m))
        return draw(arrays(dtype=np.float64, shape=(n, m), elements=reasonable_floats))

    @st.composite
    def matched_arrays(draw, *, min_n=2, max_n=100, min_m=1, max_m=10):
        """Generate a pair of 2D arrays with matching shapes (for predictions/observations)."""
        n = draw(st.integers(min_value=min_n, max_value=max_n))
        m = draw(st.integers(min_value=min_m, max_value=max_m))
        shape = (n, m)
        a = draw(arrays(dtype=np.float64, shape=shape, elements=reasonable_floats))
        b = draw(arrays(dtype=np.float64, shape=shape, elements=reasonable_floats))
        return a, b
