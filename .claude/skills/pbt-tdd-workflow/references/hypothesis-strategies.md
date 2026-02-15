# Hypothesis Strategies & Configuration

---

## When to Extract Shared Strategies

Strategies start inline in test files. Extract to `tests/strategies.py` when:

- **Used in 2+ test files** — DRY applies to test data generation too
- **Complex setup** — if a strategy needs >3 lines of construction, extract for readability
- **Domain concept** — if a strategy represents a reusable domain concept (e.g., "valid email", "matched array pair"), it deserves a name

Do NOT pre-create `tests/strategies.py` with anticipated strategies. Let them emerge from refactoring (Step 4 of the TDD cycle).

---

## Strategy Construction: Simplest to Most Complex

Prefer the simplest approach that works.

### 1. Built-in strategies with parameters

```python
st.integers(min_value=1, max_value=100)
st.text(min_size=1, max_size=50)
st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
st.lists(st.integers(), min_size=1, max_size=20)
```

### 2. `.map()` for simple transforms

```python
upper_text = st.text(min_size=1).map(str.upper)
pos_decimal = st.floats(min_value=0.01, max_value=1e6).map(Decimal)
```

### 3. `.filter()` for simple constraints (use sparingly)

```python
# Only if <~30% of generated values are rejected
even_ints = st.integers().filter(lambda x: x % 2 == 0)
# Prefer: st.integers().map(lambda x: x * 2)  — zero rejection rate
```

### 4. `st.one_of()` for union types

```python
st.one_of(st.none(), st.integers(), st.text())
```

### 5. `@st.composite` for dependent values

Use when one drawn value determines constraints on the next.

```python
@st.composite
def matched_pair(draw, *, min_len=1, max_len=50):
    """Two lists of the same length (dependent sizes)."""
    n = draw(st.integers(min_value=min_len, max_value=max_len))
    xs = draw(st.lists(st.integers(), min_size=n, max_size=n))
    ys = draw(st.lists(st.integers(), min_size=n, max_size=n))
    return xs, ys
```

### 6. `hypothesis.extra` modules for specialized types

```python
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import columns, data_frames
```

---

## Common `@st.composite` Patterns

### Dependent dimensions

```python
@st.composite
def matrix(draw, *, min_rows=1, max_rows=20, min_cols=1, max_cols=10):
    rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    cols = draw(st.integers(min_value=min_cols, max_value=max_cols))
    return draw(arrays(dtype=np.float64, shape=(rows, cols),
                       elements=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False)))
```

### Constrained combinations

```python
@st.composite
def date_range(draw):
    start = draw(st.dates())
    delta = draw(st.timedeltas(min_value=timedelta(days=1), max_value=timedelta(days=365)))
    return start, start + delta
```

### Parameterized constraints

```python
@st.composite
def bounded_list(draw, *, min_sum=0, max_sum=100):
    xs = draw(st.lists(st.integers(min_value=0, max_value=max_sum), min_size=1, max_size=10))
    assume(min_sum <= sum(xs) <= max_sum)
    return xs
```

---

## `tests/conftest.py` Template

```python
"""Shared pytest configuration and hypothesis profiles."""

from hypothesis import HealthCheck, Verbosity, settings

# --- Hypothesis profiles ---

# dev: fast feedback during development
settings.register_profile(
    "dev",
    max_examples=10,
    suppress_health_check=[HealthCheck.too_slow],
)

# ci: thorough checking
settings.register_profile(
    "ci",
    max_examples=500,
    deadline=None,
)

# debug: verbose output for investigating failures
settings.register_profile(
    "debug",
    max_examples=10,
    verbosity=Verbosity.verbose,
    suppress_health_check=[HealthCheck.too_slow],
)

# Default to dev profile for fast iteration
settings.load_profile("dev")
```

Switch profiles:

```bash
uv run pytest --hypothesis-profile=ci
uv run pytest --hypothesis-profile=debug
```

---

## `pyproject.toml` Additions

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-x --tb=short"
```

---

## Common Pitfalls

### NaN / Inf in generated floats

Always use `allow_nan=False, allow_infinity=False` in float strategies. NaN breaks comparisons silently; Inf causes overflow in arithmetic.

```python
# Bad: default allows NaN and Inf
st.floats()

# Good: explicit bounds
st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
```

### Absolute vs relative tolerance

Absolute tolerance (`a <= b + 1e-10`) fails when values are large — `1e-10` vanishes into float64 rounding at magnitudes above `1e6`. Use relative tolerance for inequality comparisons:

```python
# Bad: fails for large values
assert mae_val <= rmse_val + 1e-10

# Good: scales with magnitude
assert mae_val <= rmse_val * (1 + 1e-10) + 1e-10
```

### Very large values

Bound element ranges to avoid overflow. `1e6` is sufficient for most tests; use tighter bounds (`1e3`) if the function involves squaring or exponentiation.

### Fixture teardown

Hypothesis calls the test function many times but **fixture teardown only runs once** (at the end). Do not use stateful fixtures with `@given`. Set up state inside the test body.

### `assume()` vs constraints

Prefer built-in strategy constraints over `assume()`:

```python
# Bad: high rejection rate
@given(st.integers())
def test_bad(x):
    assume(1 <= x <= 100)

# Good: generates valid data directly
@given(st.integers(min_value=1, max_value=100))
def test_good(x):
    ...
```

### `@given` with named arguments

Always use named arguments to avoid confusion with pytest fixtures:

```python
# Good
@given(data=matched_arrays())
def test_metric(data):
    ...

# Bad: positional can confuse pytest fixture injection
@given(matched_arrays())
def test_metric(data):
    ...
```

---

## Key Imports

```python
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st
# For numpy:
# from hypothesis.extra.numpy import arrays
# For pandas:
# from hypothesis.extra.pandas import columns, data_frames
```
