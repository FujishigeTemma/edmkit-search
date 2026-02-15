---
name: pbt-tdd-workflow
description: >
  Strict TDD workflow with property-based testing using pytest and hypothesis.
  Use when writing new tests, updating tests, or when asked to test code. Enforces t_wada-style
  Red-Green-Refactor discipline: one test at a time, confirm failure before implementation.
  Covers property identification, hypothesis strategy design, and
  test design techniques (boundary value analysis, equivalence partitioning).
---

# PBT + TDD Workflow

Strict test-driven development with property-based testing. Each test follows the Red-Green-Refactor cycle one at a time. Never batch-write tests.

## Prerequisites

Before writing the first test, ensure this infrastructure exists. Read [references/hypothesis-strategies.md](references/hypothesis-strategies.md) for complete templates.

1. `tests/` directory exists
2. `tests/__init__.py` exists (empty)
3. `tests/conftest.py` with hypothesis profiles (dev/ci/debug)
4. `pyproject.toml` has `[tool.pytest.ini_options]` with `testpaths = ["tests"]`

Create any missing files from the templates in [references/hypothesis-strategies.md](references/hypothesis-strategies.md).

## The TDD Cycle

**STRICT RULE: Complete one full cycle before starting the next. Never write multiple tests at once.**

### Step 1: Identify ONE Property

Pick the next property to test. Use the decision tree below to identify it, and consult [references/property-catalog.md](references/property-catalog.md) for patterns with concrete examples.

### Step 2: RED — Write ONE Failing Test

Write a single test function. Run it:

```bash
uv run pytest tests/test_<module>.py::<test_name> -x
```

**Confirm it FAILS.** If it passes, the test is not testing anything new — rethink.
If it fails for the wrong reason (ImportError, SyntaxError), fix the test first.

### Step 3: GREEN — Minimum Code to Pass

Write the minimum implementation to make the test pass. Run:

```bash
uv run pytest tests/test_<module>.py::<test_name> -x
```

**Confirm it PASSES.** Do not add extra logic beyond what the test requires.

### Step 4: REFACTOR

Clean up code and tests while keeping everything green:

```bash
uv run pytest -x
```

Look for: duplicated setup, unclear names, opportunities to extract helpers.
When a strategy is used in 2+ test files, extract it to `tests/strategies.py` (see [references/hypothesis-strategies.md](references/hypothesis-strategies.md) for guidance).

### Property Weakening

Hypothesis will find inputs where the mathematical ideal breaks down due to floating-point. This is expected and valuable. When a test fails on a degenerate input:

1. **Understand why** — trace the floating-point computation to see where precision is lost
2. **Weaken the property** — test a weaker but universally correct property instead
3. **Test the ideal separately** — use a well-conditioned strategy to test the stronger property

**Caution**: Never weaken a property just to make a test pass. Weakening is only justified by an inherent limitation (e.g., floating-point precision loss). When you do weaken, always leave a comment explaining **what was weakened** and **why it is acceptable**.

```python
@given(arr=arrays_2d(min_n=3))
def test_self_correlation_non_negative(self, arr):
    # Weakened: mean_rho(x, x) == 1 ideally, but subnormal inputs cause
    # catastrophic cancellation in mean-centering, producing arbitrary
    # correlation values. Non-negativity still holds because cov(x,x) >= 0.
    result = mean_rho(arr, arr)
    assert np.all(result >= -1e-10)
```

### Step 5: Repeat

Return to Step 1. Pick the next property.

## Property Identification

When deciding what to test next, follow this order:

```
Function under test
├─ Start here         → Does not crash on all valid inputs (robustness)
├─ Has an inverse?    → Round-trip: decode(encode(x)) == x
├─ Has guard clauses? → Validation: invalid input raises documented exception
├─ Produces output?   → Invariants: shape/type/range preserved
├─ Obeys math laws?   → Algebraic: identity, inverse, commutativity, associativity
├─ Normalizes input?  → Idempotence: f(f(x)) == f(x)
├─ Multiple variants? → Ordering: known inequalities between variants
├─ Has a simpler ver? → Test oracle: compare against reference implementation
├─ Hard to check?     → Metamorphic: relate outputs for related inputs
└─ Orchestrates deps? → Dependency injection: inject stubs, test wiring
```

See [references/property-catalog.md](references/property-catalog.md) for complete patterns with code examples.

## Evaluating Bug Detection Power

Before writing a test, answer: **"What realistic bug would make this test fail?"** If you cannot name a concrete mistake that violates the property, the test has low detection power and should be skipped. See also [references/property-catalog.md](references/property-catalog.md) for pitfalls annotated on each pattern.

### Tautological Oracles

A test oracle must use an **independent** implementation. If the oracle reproduces the same formula as the production code, it verifies nothing — both will share the same bug.

```python
# BAD — oracle is the implementation copy-pasted
expected = np.sqrt(np.mean((pred - obs) ** 2, axis=0))
np.testing.assert_allclose(rmse(pred, obs), expected)

# GOOD — oracle uses a different library/algorithm
from sklearn.metrics import mean_squared_error
expected = np.sqrt(mean_squared_error(obs, pred, multioutput="raw_values"))
np.testing.assert_allclose(rmse(pred, obs), expected)
```

### Implementation-Evident Properties

A property that is immediately obvious from reading the implementation has low detection power. The same mathematical property can be worth testing or not, depending on how the function is implemented.

Example: translation invariance `f(p+c, o+c) = f(p, o)`.

- **Self-evident**: `rmse` computes `(predictions - observations)` first — the shift cancels by inspection. No realistic bug would break this.
- **Non-trivial**: `mean_rho` centers each array independently (`pred - pred.mean()`). A bug in centering could break shift invariance. Worth testing.

Similarly, `f(x, x) = 0`:

- **Self-evident**: `rmse(x, x)` → `sqrt(mean((x-x)^2))` = `sqrt(0)` = 0 by inspection.
- **Non-trivial**: A function that normalizes or divides internally might not handle the `diff = 0` edge case correctly.

Before writing a property test, read the implementation and ask: **"Could a realistic mistake in this code violate this property?"** If the property holds by trivial algebraic cancellation visible in the code, skip it.

### Prefer Stronger Properties, Consolidate Weaker Ones

Prefer tests that verify broader properties with wider bug detection coverage. When a single stronger test subsumes multiple weaker tests, actively replace them.

Example: **scale equivariance** `f(a·p, a·o) = |a|·f(p, o)` detects:
- forgot `abs` in MAE → gives `a` instead of `|a|` when `a < 0`
- forgot `sqrt` in RMSE → gives `a^2` instead of `|a|`

This one test covers the same bugs as **non-negativity** (`f >= 0`) and **MAE <= RMSE** (Jensen's inequality) combined. In this case, scale equivariance alone should remain and the other two should be removed.

When a test covers multiple properties, document which properties it verifies and the reasoning in comments:

```python
@given(data=matched_arrays(), a=reasonable_floats)
def test_error_metrics_scale_equivariant(metric, data, a):
    """f(a·p, a·o) = |a|·f(p, o).

    This property also verifies:
    - Non-negativity: setting a = -1 shows f(p, o) = f(-p, -o) = |-1|·f(p, o),
      meaning f is non-negative whenever it equals its own absolute-scaled value.
    - Correct use of abs (MAE): if abs were missing, a < 0 would give
      a·f(p,o) != |a|·f(p,o).
    - Correct use of sqrt (RMSE): if sqrt were missing, scaling would give
      a^2·f(p,o) != |a|·f(p,o).
    """
```

### Trivial Functions

For functions where the implementation is trivially short (e.g., `return -f(x)`), property-based tests are overkill. A single example-based test suffices, or the function can be covered indirectly through callers.

## Test Conventions

### File naming

```
tests/
├── conftest.py
├── strategies.py     # created during refactoring when strategies are shared
└── test_<module>.py  # one test file per source module
```

### Import pattern

```python
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mypackage.module import function_under_test
# When shared strategies exist:
# from tests.strategies import my_strategy
```

### @given with named arguments

Always use named keyword arguments to avoid confusion with pytest fixtures:

```python
@given(data=matched_arrays())
def test_metric_shape(data):
    preds, obs = data
    ...
```

### @example for edge cases

Pin known edge cases alongside property-based generation:

```python
@example(x="")     # empty string
@example(x="\x00") # null byte
@given(x=st.text())
def test_handles_edge_cases(x):
    ...
```

### Assertions

```python
# Standard
assert result == expected
assert isinstance(result, list)
assert len(result) == len(input_data)

# Floating-point (pytest)
assert result == pytest.approx(expected, rel=1e-6)

# Numpy (when applicable)
# np.testing.assert_array_almost_equal(result, expected)
# assert np.all(np.isfinite(result))
```

## Testing Order

Test bottom-up by dependency. Start with modules that have no internal dependencies (pure functions, leaf modules), then work up to modules that depend on them. Each layer's dependencies should be tested before you test that layer.

## Anti-Patterns

| Do NOT | Instead |
|--------|---------|
| Write many tests, then implement | One test → Red → Green → Refactor → repeat |
| Skip the RED step | Always confirm the test fails first |
| Use `assume()` for constrainable data | Use strategy parameters: `min_value`, `max_value`, `min_size` |
| Test implementation details | Test properties and contracts from docstrings |
| Replicate the implementation's internal logic in test conditions | Test weaker but universally correct properties |
| Use absolute tolerance for inequality comparisons | Use relative tolerance: `a <= b * (1 + eps) + eps` |
| Use stateful pytest fixtures with `@given` | Set up state inside the test body |
| Generate floats with NaN/Inf | Use `allow_nan=False, allow_infinity=False` |
| Write an oracle that copies the implementation formula | Use an independent reference (different library or algorithm) |
| Test a property self-evident from reading the implementation | Test properties that constrain the function's non-obvious behavior |
| Add a test whose bug detection overlaps with an existing test | Check what bugs each test uniquely catches |
| Write PBT for trivial one-line functions | Use a single example-based test, or cover indirectly through callers |
| Only run dev profile | Validate with `--hypothesis-profile=ci` before finishing |
