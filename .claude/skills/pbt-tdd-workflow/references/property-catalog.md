# Property Catalog

Property patterns for property-based testing, organized from simplest to most sophisticated. Each category includes the principle, a general example, and optionally a domain-specific example.

---

## 1. Does Not Crash / Robustness

The function handles all valid inputs without raising unexpected exceptions. The simplest property and a good starting point for any function.

```python
@given(x=st.text())
def test_parse_does_not_crash(x):
    try:
        parse(x)
    except ParseError:
        pass  # Expected for invalid input; no other exception allowed
```

**Pattern**: For any valid input from the type signature, the function either returns a result or raises a documented exception — never an unexpected crash.

---

## 2. Round-Trip / Encode-Decode

`decode(encode(x)) == x`. Reversible transformations preserve information.

```python
@given(data=st.dictionaries(st.text(), st.integers()))
def test_json_round_trip(data):
    assert json.loads(json.dumps(data)) == data
```

```python
# Scientific: normalize/denormalize
@given(arr=arrays(dtype=np.float64, shape=(5,), elements=reasonable_floats))
def test_normalize_round_trip(arr):
    normed, params = normalize(arr)
    np.testing.assert_array_almost_equal(denormalize(normed, params), arr)
```

**Pattern**: If `f` has an inverse `g`, then `g(f(x)) == x`. For lossy transforms, test the weaker `f(g(f(x))) == f(x)`.

---

## 3. Invariants / Some Things Never Change

Certain properties of the output are preserved regardless of input values.

### 3a. Structural / Shape Invariants

```python
@given(xs=st.lists(st.integers()))
def test_sorted_preserves_length(xs):
    assert len(sorted(xs)) == len(xs)

@given(xs=st.lists(st.integers()))
def test_sorted_preserves_elements(xs):
    assert sorted(sorted(xs)) == sorted(xs)  # same multiset
```

```python
# Scientific: output shape matches contract
@given(data=matched_arrays())
def test_metric_output_shape(data):
    preds, obs = data  # shape (N, M)
    result = mean_rho(preds, obs)
    assert result.shape == (preds.shape[1],)
```

### 3b. Range / Bounds Invariants

```python
@given(xs=st.lists(st.integers(), min_size=1))
def test_clamp_in_range(xs):
    result = [clamp(x, lo=0, hi=100) for x in xs]
    assert all(0 <= v <= 100 for v in result)
```

```python
# Scientific: correlation ∈ [-1, 1], error metrics >= 0
@given(data=matched_arrays(min_n=3))
def test_correlation_bounded(data):
    preds, obs = data
    result = mean_rho(preds, obs)
    assert np.all((-1 - 1e-10 <= result) & (result <= 1 + 1e-10))
```

**Pattern**: Identify what must always be true about the output's structure, size, type, or value range — these hold for ALL valid inputs.

---

## 4. Algebraic Properties

Functions obey algebraic laws: identity, inverse, commutativity, associativity, distributivity, involution.

```python
# Identity element
@given(xs=st.lists(st.integers()))
def test_concat_identity(xs):
    assert xs + [] == xs

# Involution (self-inverse)
@given(s=st.text())
def test_reverse_involution(s):
    assert s[::-1][::-1] == s

# Commutativity
@given(a=st.frozensets(st.integers()), b=st.frozensets(st.integers()))
def test_union_commutative(a, b):
    assert a | b == b | a

# Associativity
@given(a=st.text(), b=st.text(), c=st.text())
def test_concat_associative(a, b, c):
    assert (a + b) + c == a + (b + c)
```

```python
# Scientific: negate is involution
@given(data=matched_arrays())
def test_negate_is_involution(data):
    preds, obs = data
    np.testing.assert_array_almost_equal(
        negate(negate(mean_rho))(preds, obs),
        mean_rho(preds, obs),
    )

# Perfect prediction: metric(x, x) = ideal score
@given(arr=arrays_2d(min_n=3))
def test_perfect_prediction_rmse_zero(arr):
    np.testing.assert_array_almost_equal(rmse(arr, arr), 0.0)
```

**Pattern**: Identity elements, inverse operations, commutativity, associativity, distributivity, involution, perfect-input behavior.

**Pitfall — Implementation-evident identities**: Before testing an algebraic identity, read the implementation and ask whether the property is self-evident from the code. For example, `rmse(x, x) = 0` is trivially `sqrt(mean((x-x)^2))` = 0 — no realistic bug would break this. On the other hand, `mean_rho(x, x) = 1` involves centering, division, and a zero-variance guard, making it a genuinely useful test.

---

## 5. Idempotence

Applying an operation twice yields the same result as once: `f(f(x)) == f(x)`.

```python
@given(s=st.text())
def test_strip_idempotent(s):
    assert s.strip().strip() == s.strip()

@given(xs=st.lists(st.integers()))
def test_deduplicate_idempotent(xs):
    assert deduplicate(deduplicate(xs)) == deduplicate(xs)
```

```python
# Scientific: coercion function
@given(arr=arrays_1d_or_2d())
def test_ensure_2d_idempotent(arr):
    once = _ensure_2d(arr)
    twice = _ensure_2d(once)
    np.testing.assert_array_equal(once, twice)
```

**Pattern**: `f(f(x)) == f(x)` for normalization, formatting, coercion, deduplication, canonicalization.

---

## 6. Ordering / Inequalities

Known mathematical or logical relationships between operations.

```python
@given(xs=st.lists(st.integers(), min_size=1))
def test_min_leq_max(xs):
    assert min(xs) <= max(xs)

@given(xs=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1))
def test_mean_between_extremes(xs):
    assert min(xs) <= statistics.mean(xs) <= max(xs)
```

```python
# Scientific: MAE <= RMSE (Jensen's inequality)
@given(data=matched_arrays(min_n=2))
def test_mae_leq_rmse(data):
    preds, obs = data
    m, r = mae(preds, obs), rmse(preds, obs)
    assert np.all(m <= r * (1 + 1e-10) + 1e-10)
```

**Pattern**: Known inequalities from definitions (Jensen's, triangle inequality, monotonicity). Use relative + absolute tolerance for floating-point comparisons.

**Pitfall — Redundancy with other properties**: Ordering tests often catch the same bugs as other property tests. For example, MAE <= RMSE (Jensen) catches "forgot `sqrt` in RMSE", but so does scale equivariance. Check that an ordering test provides unique detection power before adding it.

---

## 7. Validation / Fail-Fast

Invalid inputs raise documented exceptions with descriptive messages.

```python
@given(port=st.integers(max_value=-1))
def test_negative_port_raises(port):
    with pytest.raises(ValueError, match="port"):
        connect(host="localhost", port=port)
```

```python
# Scientific: shape mismatch
@given(
    n=st.integers(min_value=2, max_value=50),
    m1=st.integers(min_value=1, max_value=5),
    m2=st.integers(min_value=1, max_value=5),
)
def test_shape_mismatch_raises(n, m1, m2):
    if m1 == m2:
        return
    a, b = np.zeros((n, m1)), np.zeros((n, m2))
    with pytest.raises(ValueError, match="Shape mismatch"):
        mean_rho(a, b)
```

**Pattern**: For each guard clause in the function, write a test that triggers it and verifies the error message content.

---

## 8. Test Oracle / Reference Implementation

Compare the function under test against a known-correct but possibly slower or simpler implementation.

```python
@given(xs=st.lists(st.integers(), min_size=1))
def test_custom_sort_matches_builtin(xs):
    assert my_sort(xs) == sorted(xs)
```

```python
# Scientific: optimized RMSE vs independent reference
@given(data=matched_arrays(min_n=3, max_n=20, max_m=3))
def test_fast_rmse_matches_sklearn(data):
    from sklearn.metrics import mean_squared_error
    preds, obs = data
    expected = np.sqrt(mean_squared_error(obs, preds, multioutput="raw_values"))
    np.testing.assert_array_almost_equal(rmse(preds, obs), expected)
```

**Pattern**: When a simpler reference implementation exists, test equivalence on random inputs. This is the most effective single property type for bug detection (Hughes, 2020).

**Pitfall — Tautological oracle**: The oracle must be **independent** of the implementation. If both use the same formula (`np.sqrt(np.mean((p - o) ** 2, axis=0))`), the test is a copy-paste that catches nothing. Use a different library, algorithm, or loop-based naive version.

---

## 9. Metamorphic Relations

When you cannot easily determine expected output, check *relationships* between outputs for related inputs.

```python
# Adding more data should not reduce search results
@given(query=st.text(min_size=1), extra=st.text(min_size=1))
def test_search_monotonic(query, extra):
    small = search(query, corpus=["hello", "world"])
    large = search(query, corpus=["hello", "world", extra])
    assert len(large) >= len(small)

# Negating input negates output
@given(data=matched_arrays())
def test_negate_inverts_sign(data):
    preds, obs = data
    np.testing.assert_array_equal(
        negate(mean_rho)(preds, obs),
        -mean_rho(preds, obs),
    )
```

**Pattern**: If the input changes in a predictable way, the output should change predictably — monotonicity, symmetry, permutation invariance, scaling. Critical for oracle-less functions (ML models, search, scientific code).

---

## 10. Commutativity / Different Paths, Same Destination

Performing the same operations in different orders produces the same result.

```python
@given(xs=st.lists(st.integers()))
def test_filter_then_sort_eq_sort_then_filter(xs):
    f = lambda x: x > 0
    assert sorted(filter(f, xs)) == list(filter(f, sorted(xs)))

@given(s=st.text(), a=st.text(), b=st.text())
def test_replace_order_independent(s, a, b):
    # When replacements don't overlap, order shouldn't matter
    assume(a not in b and b not in a and a != "" and b != "")
    r1 = s.replace(a, "").replace(b, "")
    r2 = s.replace(b, "").replace(a, "")
    assert r1 == r2
```

**Pattern**: If two operations are independent, their composition should commute. Applies to batch vs. incremental processing, parallel execution, map-then-filter vs. filter-then-map.

---

## 11. Hard to Compute, Easy to Verify

Computing a result is complex, but checking it is straightforward.

```python
import math
from functools import reduce
from operator import mul

@given(n=st.integers(min_value=2, max_value=10_000))
def test_prime_factors_multiply_back(n):
    factors = prime_factors(n)
    assert reduce(mul, factors) == n
    assert all(is_prime(f) for f in factors)
```

**Pattern**: Verify the *result* satisfies a checkable constraint, rather than replicating the computation. Useful for optimization, search, constraint satisfaction, factoring, pathfinding.

---

## 12. Dependency Injection / Wiring

Inject trivial callbacks to isolate orchestration logic from the behavior of dependencies.

```python
def test_retry_calls_function_n_times():
    call_count = 0
    def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError
        return "ok"
    assert retry(flaky, max_attempts=5) == "ok"
    assert call_count == 3
```

```python
# Scientific: inject trivial predict function
def test_prediction_skill_with_identity():
    manifold = np.random.randn(50, 3)
    target = np.random.randn(50, 2)
    def dummy_predict(X_lib, Y_lib, X_query):
        return np.zeros((len(X_query), Y_lib.shape[1]))
    score, per_target, preds = prediction_skill(
        manifold, target, np.arange(30), np.arange(30, 50),
        predict=dummy_predict, metric=mean_rho,
    )
    assert per_target.shape == (2,)
```

**Pattern**: Replace injected dependencies with lambdas/stubs. Test the orchestration and wiring, not the dependency's behavior.
