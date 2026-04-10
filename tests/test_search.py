from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.spatial.distance import cdist

from functools import partial

from edmkit.metrics import mae
from edmkit.search import Dataset, Selection, Step, collect
from edmkit.search import anneal, beam, geometric_cooling, greedy
from edmkit.search import (
    greedy_complementary_folds,
    greedy_complementary_timepoints,
    mean_abs_error_per_sample,
    mean_negative_correlation_contribution_per_sample,
    mean_squared_error_per_sample,
    softmax_loss_weight,
    softmax_weight,
)
from edmkit.search.common import negate, score_subset, score_subset_per_fold
from edmkit.splits import Fold, sliding_folds, temporal_fold

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_dataset(
    N: int = 50,
    M: int = 5,
    D: int = 1,
    *,
    seed: int = 42,
) -> Dataset:
    """Create a deterministic dataset for testing."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, M))
    Y = rng.standard_normal((N, D))
    return Dataset(X=X, Y=Y)


def dummy_predict(
    X: np.ndarray,
    Y: np.ndarray,
    Q: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Predict by nearest-neighbor (1-NN) for testing."""
    dists = cdist(Q, X)
    nearest = np.argmin(dists, axis=1)
    return Y[nearest]


def mean_abs_corr(predictions: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """Mean absolute correlation across dimensions."""
    M = predictions.shape[1]
    total = 0.0
    for m in range(M):
        p = predictions[:, m]
        o = observations[:, m]
        if p.std() < 1e-12 or o.std() < 1e-12:
            continue
        total += abs(float(np.corrcoef(p, o)[0, 1]))
    return np.asarray(total / max(M, 1))


# ---------------------------------------------------------------------------
# Cheap predict/metric for property-based tests
# ---------------------------------------------------------------------------


def colsum_predict(
    X: np.ndarray,
    Y: np.ndarray,
    Q: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Cheap predict: column sum of query, repeated across Y dims."""
    return Q.sum(axis=1, keepdims=True).repeat(Y.shape[1], axis=1)


neg_mae = negate(mae)


# ---------------------------------------------------------------------------
# Hypothesis strategy
# ---------------------------------------------------------------------------

elems = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)


@st.composite
def search_inputs_impl(draw, *, min_m=2, max_m=8, min_n=4, max_n=30):
    """Generate (Dataset, max_dim) pairs for search tests."""
    M = draw(st.integers(min_value=min_m, max_value=max_m))
    N = draw(st.integers(min_value=min_n, max_value=max_n))
    max_dim = draw(st.integers(min_value=1, max_value=M))
    X = draw(arrays(np.float64, (N, M), elements=elems))
    Y = draw(arrays(np.float64, (N, 1), elements=elems))
    return Dataset(X=X, Y=Y), max_dim


def search_inputs(
    *, min_m: int = 2, max_m: int = 8, min_n: int = 4, max_n: int = 30
) -> st.SearchStrategy[tuple[Dataset, int]]:
    """Generate (Dataset, max_dim) pairs for search tests."""
    return search_inputs_impl(min_m=min_m, max_m=max_m, min_n=min_n, max_n=max_n)  # type: ignore[missing-argument]


# ---------------------------------------------------------------------------
# 1. Structural invariants (property-based)
# ---------------------------------------------------------------------------


def run(algo: str, ds: Dataset, max_dim: int) -> list[Step]:
    """Run a search algorithm with cheap predict/metric."""
    if algo == "greedy":
        return list(
            greedy(
                ds,
                ds,
                predict=colsum_predict,
                metric=neg_mae,
                max_dim=max_dim,
                threshold=-float("inf"),
            )
        )
    elif algo == "beam":
        return list(
            beam(
                ds,
                ds,
                predict=colsum_predict,
                metric=neg_mae,
                max_dim=max_dim,
                beam_width=2,
                threshold=-float("inf"),
            )
        )
    elif algo == "anneal":
        return list(
            anneal(
                ds,
                ds,
                predict=colsum_predict,
                metric=neg_mae,
                n_steps=20,
                max_dim=max_dim,
                rng=np.random.default_rng(0),
            )
        )
    elif algo == "complementary":
        T = ds.X.shape[0]
        ts = max(T // 3, 1)
        vs = max(T // 3, 1)
        split = partial(sliding_folds, train_size=ts, validation_size=vs)
        return list(
            greedy_complementary_folds(
                ds,
                predict=colsum_predict,
                metric=neg_mae,
                split=split,
                max_dim=max_dim,
                threshold=-float("inf"),
            )
        )
    raise ValueError(algo)


class TestStructuralInvariants:
    """Property-based tests for invariants that hold across all algorithms."""

    @given(data=search_inputs())
    @pytest.mark.parametrize("algo", ["greedy", "beam", "anneal", "complementary"])
    def test_index_is_last_selected(self, algo: str, data: tuple[Dataset, int]):
        """step.index == step.selected[-1] for all algorithms."""
        ds, max_dim = data
        for step in run(algo, ds, max_dim):
            assert step.index == step.selected[-1]

    @given(data=search_inputs())
    @pytest.mark.parametrize("algo", ["greedy", "beam", "anneal", "complementary"])
    def test_no_duplicate_indices(self, algo: str, data: tuple[Dataset, int]):
        """No duplicate indices in step.selected."""
        ds, max_dim = data
        for step in run(algo, ds, max_dim):
            assert len(step.selected) == len(set(step.selected))

    @given(data=search_inputs())
    @pytest.mark.parametrize("algo", ["greedy", "beam", "anneal", "complementary"])
    def test_indices_in_range(self, algo: str, data: tuple[Dataset, int]):
        """All indices in [0, M)."""
        ds, max_dim = data
        M = ds.X.shape[1]
        for step in run(algo, ds, max_dim):
            for idx in step.selected:
                assert 0 <= idx < M

    @given(data=search_inputs())
    @pytest.mark.parametrize("algo", ["greedy", "beam", "anneal", "complementary"])
    def test_selected_within_max_dim(self, algo: str, data: tuple[Dataset, int]):
        """len(step.selected) <= max_dim."""
        ds, max_dim = data
        for step in run(algo, ds, max_dim):
            assert len(step.selected) <= max_dim

    @given(data=search_inputs())
    @pytest.mark.parametrize("algo", ["greedy", "beam", "anneal", "complementary"])
    def test_score_is_finite(self, algo: str, data: tuple[Dataset, int]):
        """step.score is finite."""
        ds, max_dim = data
        for step in run(algo, ds, max_dim):
            assert math.isfinite(step.score)

    @given(data=search_inputs())
    @pytest.mark.parametrize("algo", ["greedy", "beam", "complementary"])
    def test_selected_grows_by_one(self, algo: str, data: tuple[Dataset, int]):
        """len(selected) grows by exactly 1 per step (greedy/beam only)."""
        ds, max_dim = data
        prev_len = 0
        for step in run(algo, ds, max_dim):
            assert len(step.selected) == prev_len + 1
            prev_len = len(step.selected)

    @given(data=search_inputs())
    @pytest.mark.parametrize("algo", ["greedy", "beam", "anneal", "complementary"])
    def test_collect_invariants(self, algo: str, data: tuple[Dataset, int]):
        """collect() length consistency and indices match last step."""
        ds, max_dim = data
        steps = run(algo, ds, max_dim)
        sel = collect(iter(steps))
        assert len(sel.indices) == len(sel.scores)
        if steps:
            assert sel.indices == list(steps[-1].selected)


# ---------------------------------------------------------------------------
# 2. Metamorphic relations (property-based)
# ---------------------------------------------------------------------------


class TestMetamorphicRelations:
    """Tests for relationships between outputs under input transformations."""

    @given(data=search_inputs(), perm_seed=st.integers(0, 2**32 - 1))
    def test_greedy_permutation_invariant(
        self, data: tuple[Dataset, int], perm_seed: int
    ):
        """Greedy achieves the same scores regardless of column order."""
        ds, max_dim = data
        rng = np.random.default_rng(perm_seed)
        perm = rng.permutation(ds.X.shape[1])

        ds_perm = Dataset(X=ds.X[:, perm], Y=ds.Y)

        orig_steps = list(
            greedy(
                ds,
                ds,
                predict=colsum_predict,
                metric=neg_mae,
                max_dim=max_dim,
                threshold=-float("inf"),
            )
        )
        perm_steps = list(
            greedy(
                ds_perm,
                ds_perm,
                predict=colsum_predict,
                metric=neg_mae,
                max_dim=max_dim,
                threshold=-float("inf"),
            )
        )

        assert len(orig_steps) == len(perm_steps)
        # Scores at each step should match (ties may break differently)
        for g, p in zip(orig_steps, perm_steps):
            np.testing.assert_allclose(g.score, p.score, rtol=1e-6)

    @given(data=search_inputs(), t_offset=st.floats(0.01, 100.0))
    def test_threshold_monotonicity(self, data: tuple[Dataset, int], t_offset: float):
        """Higher threshold -> fewer or equal variables selected."""
        ds, max_dim = data
        lo = -float("inf")
        hi = t_offset  # positive threshold

        steps_lo = list(
            greedy(
                ds,
                ds,
                predict=colsum_predict,
                metric=neg_mae,
                max_dim=max_dim,
                threshold=lo,
            )
        )
        steps_hi = list(
            greedy(
                ds,
                ds,
                predict=colsum_predict,
                metric=neg_mae,
                max_dim=max_dim,
                threshold=hi,
            )
        )
        assert len(steps_lo) >= len(steps_hi)

    @given(
        T_start=st.floats(0.1, 100.0, allow_nan=False, allow_infinity=False),
        T_end=st.floats(0.001, 10.0, allow_nan=False, allow_infinity=False),
        n_steps=st.integers(2, 100),
    )
    def test_geometric_cooling_bounded(
        self, T_start: float, T_end: float, n_steps: int
    ):
        """geometric_cooling output is bounded by [T_end, T_start]."""
        assume(T_start > T_end)
        sched = geometric_cooling(T_start=T_start, T_end=T_end)
        for step in range(n_steps):
            T = sched(step, n_steps)
            assert T_end <= T + 1e-10  # small tolerance
            assert T <= T_start + 1e-10


# ---------------------------------------------------------------------------
# 3. Oracle comparisons (property-based)
# ---------------------------------------------------------------------------


class TestOracleComparisons:
    """beam_width=1 should produce same results as greedy."""

    @given(data=search_inputs())
    def test_beam_width_1_equals_greedy(self, data: tuple[Dataset, int]):
        """beam(beam_width=1) == greedy for all valid inputs."""
        ds, max_dim = data
        greedy_steps = list(
            greedy(
                ds,
                ds,
                predict=colsum_predict,
                metric=neg_mae,
                max_dim=max_dim,
                threshold=-float("inf"),
            )
        )
        beam_steps = list(
            beam(
                ds,
                ds,
                predict=colsum_predict,
                metric=neg_mae,
                max_dim=max_dim,
                beam_width=1,
                threshold=-float("inf"),
            )
        )
        assert len(greedy_steps) == len(beam_steps)
        for g, b in zip(greedy_steps, beam_steps):
            assert g.index == b.index
            np.testing.assert_allclose(g.score, b.score, rtol=1e-10)
            assert g.selected == b.selected


# ---------------------------------------------------------------------------
# 4. Validation errors (property-based)
# ---------------------------------------------------------------------------


class TestValidationPBT:
    """Property-based tests for validation error handling."""

    @given(
        M=st.integers(1, 8),
        excess=st.integers(1, 10),
    )
    @pytest.mark.parametrize("algo", ["greedy", "beam", "anneal", "complementary"])
    def test_max_dim_exceeds_M(self, algo: str, M: int, excess: int):
        """max_dim > M raises ValueError for all algorithms."""
        ds = make_dataset(M=M)
        with pytest.raises(ValueError, match="max_dim"):
            run(algo, ds, max_dim=M + excess)

    @given(width=st.integers(max_value=0))
    def test_beam_width_below_1(self, width: int):
        """beam_width < 1 raises ValueError."""
        ds = make_dataset()
        with pytest.raises(ValueError, match="beam_width"):
            list(
                beam(
                    ds,
                    ds,
                    predict=colsum_predict,
                    metric=neg_mae,
                    max_dim=3,
                    beam_width=width,
                )
            )


# ---------------------------------------------------------------------------
# 5. Step invariants (deterministic)
# ---------------------------------------------------------------------------


class TestStepInvariants:
    """Step structural properties — deterministic tests with 1-NN predict."""

    def test_greedy_index_is_last_selected(self):
        """step.index == step.selected[-1] for greedy."""
        ds = make_dataset()
        for step in greedy(
            ds, ds, predict=dummy_predict, metric=mean_abs_corr, max_dim=3
        ):
            assert step.index == step.selected[-1]

    def test_beam_index_is_last_selected(self):
        """step.index == step.selected[-1] for beam."""
        ds = make_dataset()
        for step in beam(
            ds,
            ds,
            predict=dummy_predict,
            metric=mean_abs_corr,
            max_dim=3,
            beam_width=2,
        ):
            assert step.index == step.selected[-1]

    def test_greedy_selected_monotonically_grows(self):
        """len(step.selected) increases by 1 each step for greedy."""
        ds = make_dataset()
        prev_len = 0
        for step in greedy(
            ds, ds, predict=dummy_predict, metric=mean_abs_corr, max_dim=3
        ):
            assert len(step.selected) == prev_len + 1
            prev_len = len(step.selected)

    def test_step_is_namedtuple(self):
        """Step is a NamedTuple with correct fields."""
        s = Step(index=0, score=0.5, selected=(0,))
        assert s.index == 0
        assert s.score == 0.5
        assert s.selected == (0,)


# ---------------------------------------------------------------------------
# 2. Greedy tests
# ---------------------------------------------------------------------------


class TestGreedy:
    def test_basic_operation(self):
        """Greedy selects at least one variable."""
        ds = make_dataset()
        steps = list(
            greedy(
                ds,
                ds,
                predict=dummy_predict,
                metric=mean_abs_corr,
                max_dim=3,
                threshold=-float("inf"),
            )
        )
        assert len(steps) >= 1
        assert len(steps) <= 3

    def test_threshold_filters(self):
        """High threshold stops selection early."""
        ds = make_dataset()
        steps = list(
            greedy(
                ds,
                ds,
                predict=dummy_predict,
                metric=mean_abs_corr,
                max_dim=5,
                threshold=999.0,
            )
        )
        assert len(steps) == 0

    def test_filter_rejects_all(self):
        """Filter that rejects everything yields no steps."""
        ds = make_dataset()

        def reject_all(x: np.ndarray, Y: np.ndarray) -> bool:
            return False

        steps = list(
            greedy(
                ds,
                ds,
                predict=dummy_predict,
                metric=mean_abs_corr,
                max_dim=3,
                filter=reject_all,
            )
        )
        assert len(steps) == 0

    def test_max_dim_validation(self):
        """max_dim > N raises ValueError."""
        ds = make_dataset(M=3)
        with pytest.raises(ValueError, match="max_dim"):
            list(
                greedy(
                    ds,
                    ds,
                    predict=dummy_predict,
                    metric=mean_abs_corr,
                    max_dim=10,
                )
            )

    def test_no_duplicate_indices(self):
        """Greedy never selects the same variable twice."""
        ds = make_dataset()
        steps = list(
            greedy(
                ds,
                ds,
                predict=dummy_predict,
                metric=mean_abs_corr,
                max_dim=5,
                threshold=-float("inf"),
            )
        )
        indices = [s.index for s in steps]
        assert len(indices) == len(set(indices))


# ---------------------------------------------------------------------------
# 3. Beam tests
# ---------------------------------------------------------------------------


class TestBeam:
    def test_beam_width_1_equals_greedy(self):
        """beam(beam_width=1) produces same results as greedy."""
        ds = make_dataset(seed=123)
        greedy_steps = list(
            greedy(
                ds,
                ds,
                predict=dummy_predict,
                metric=mean_abs_corr,
                max_dim=3,
                threshold=-float("inf"),
            )
        )
        beam_steps = list(
            beam(
                ds,
                ds,
                predict=dummy_predict,
                metric=mean_abs_corr,
                max_dim=3,
                beam_width=1,
                threshold=-float("inf"),
            )
        )
        assert len(greedy_steps) == len(beam_steps)
        for g, b in zip(greedy_steps, beam_steps):
            assert g.index == b.index
            np.testing.assert_allclose(g.score, b.score, rtol=1e-10)
            assert g.selected == b.selected

    def test_beam_selected_consistency(self):
        """Each step's selected is a valid, non-overlapping set."""
        ds = make_dataset()
        for step in beam(
            ds,
            ds,
            predict=dummy_predict,
            metric=mean_abs_corr,
            max_dim=3,
            beam_width=2,
            threshold=-float("inf"),
        ):
            # No duplicates in selected
            assert len(step.selected) == len(set(step.selected))

    def test_beam_filter(self):
        """Filter is respected by beam search."""
        ds = make_dataset(M=4)

        blocked = {0, 1}

        def allow_subset(x: np.ndarray, Y: np.ndarray) -> bool:
            # Identify column by content match
            for i in blocked:
                if np.array_equal(x, ds.X[:, i]):
                    return False
            return True

        steps = list(
            beam(
                ds,
                ds,
                predict=dummy_predict,
                metric=mean_abs_corr,
                max_dim=3,
                beam_width=2,
                filter=allow_subset,
                threshold=-float("inf"),
            )
        )
        for step in steps:
            for idx in step.selected:
                assert idx not in blocked

    def test_beam_width_validation(self):
        """beam_width < 1 raises ValueError."""
        ds = make_dataset()
        with pytest.raises(ValueError, match="beam_width"):
            list(
                beam(
                    ds,
                    ds,
                    predict=dummy_predict,
                    metric=mean_abs_corr,
                    max_dim=3,
                    beam_width=0,
                )
            )


# ---------------------------------------------------------------------------
# 4. Anneal tests
# ---------------------------------------------------------------------------


class TestAnneal:
    def test_deterministic_with_seed(self):
        """Same seed produces same results."""
        ds = make_dataset()
        steps1 = list(
            anneal(
                ds,
                ds,
                predict=dummy_predict,
                metric=mean_abs_corr,
                n_steps=50,
                max_dim=3,
                rng=np.random.default_rng(42),
            )
        )
        steps2 = list(
            anneal(
                ds,
                ds,
                predict=dummy_predict,
                metric=mean_abs_corr,
                n_steps=50,
                max_dim=3,
                rng=np.random.default_rng(42),
            )
        )
        assert len(steps1) == len(steps2)
        for s1, s2 in zip(steps1, steps2):
            assert s1.index == s2.index
            np.testing.assert_allclose(s1.score, s2.score, rtol=1e-10)

    def test_max_dim_constraint(self):
        """Anneal respects max_dim."""
        ds = make_dataset()
        steps = list(
            anneal(
                ds,
                ds,
                predict=dummy_predict,
                metric=mean_abs_corr,
                n_steps=50,
                max_dim=2,
                rng=np.random.default_rng(0),
            )
        )
        assert len(steps) <= 2
        if steps:
            assert len(steps[-1].selected) <= 2

    def test_filter_respected(self):
        """Anneal respects filter."""
        ds = make_dataset(M=4)
        blocked = {0}

        def allow_subset(x: np.ndarray, Y: np.ndarray) -> bool:
            for i in blocked:
                if np.array_equal(x, ds.X[:, i]):
                    return False
            return True

        steps = list(
            anneal(
                ds,
                ds,
                predict=dummy_predict,
                metric=mean_abs_corr,
                n_steps=50,
                max_dim=3,
                filter=allow_subset,
                rng=np.random.default_rng(0),
            )
        )
        for step in steps:
            for idx in step.selected:
                assert idx not in blocked

    def test_custom_schedule(self):
        """Custom ScheduleFn is used."""
        ds = make_dataset()
        calls: list[tuple[int, int]] = []

        def tracking_schedule(step: int, n_steps: int) -> float:
            calls.append((step, n_steps))
            return 1.0  # constant temperature

        list(
            anneal(
                ds,
                ds,
                predict=dummy_predict,
                metric=mean_abs_corr,
                n_steps=20,
                max_dim=2,
                schedule=tracking_schedule,
                rng=np.random.default_rng(0),
            )
        )
        assert len(calls) == 20
        assert all(n == 20 for _, n in calls)

    def test_anneal_yields_steps(self):
        """Anneal yields at least one step with low threshold."""
        ds = make_dataset()
        steps = list(
            anneal(
                ds,
                ds,
                predict=dummy_predict,
                metric=mean_abs_corr,
                n_steps=50,
                max_dim=3,
                rng=np.random.default_rng(0),
            )
        )
        assert len(steps) >= 1


# ---------------------------------------------------------------------------
# 5. collect tests
# ---------------------------------------------------------------------------


class TestCollect:
    def test_collect_empty(self):
        """collect of empty iterator returns empty Selection."""
        result = collect(iter([]))
        assert result.indices == []
        assert result.scores == []

    def test_collect_uses_last_selected(self):
        """Selection.indices == list(last_step.selected)."""
        steps = [
            Step(index=2, score=0.5, selected=(2,)),
            Step(index=5, score=0.7, selected=(2, 5)),
            Step(index=1, score=0.8, selected=(2, 5, 1)),
        ]
        result = collect(steps)
        assert result.indices == [2, 5, 1]
        assert result.scores == [0.5, 0.7, 0.8]

    def test_collect_from_greedy(self):
        """collect(greedy(...)) returns valid Selection."""
        ds = make_dataset()
        result = collect(
            greedy(
                ds,
                ds,
                predict=dummy_predict,
                metric=mean_abs_corr,
                max_dim=3,
                threshold=-float("inf"),
            )
        )
        assert isinstance(result, Selection)
        assert len(result.indices) == len(result.scores)


# ---------------------------------------------------------------------------
# 6. geometric_cooling tests
# ---------------------------------------------------------------------------


class TestGeometricCooling:
    def test_endpoints(self):
        """Schedule hits T_start at step=0 and T_end at step=n_steps-1."""
        sched = geometric_cooling(T_start=10.0, T_end=0.1)
        np.testing.assert_allclose(sched(0, 100), 10.0, rtol=1e-10)
        np.testing.assert_allclose(sched(99, 100), 0.1, rtol=1e-10)

    def test_monotonically_decreasing(self):
        """Temperature decreases monotonically."""
        sched = geometric_cooling(T_start=5.0, T_end=0.01)
        temps = [sched(i, 50) for i in range(50)]
        for i in range(1, len(temps)):
            assert temps[i] < temps[i - 1]

    def test_invalid_params(self):
        """Invalid parameters raise ValueError."""
        with pytest.raises(ValueError, match="T_start"):
            geometric_cooling(T_start=-1.0)
        with pytest.raises(ValueError, match="T_end"):
            geometric_cooling(T_end=-1.0)
        with pytest.raises(ValueError, match="T_start must be > T_end"):
            geometric_cooling(T_start=0.1, T_end=1.0)


# ---------------------------------------------------------------------------
# 7. score_subset_per_fold tests
# ---------------------------------------------------------------------------


class TestScoreSubsetPerFold:
    def test_shape(self):
        """Returns array of shape (n_folds,)."""
        ds = make_dataset(N=50, M=5)
        folds = sliding_folds(50, train_size=15, validation_size=10)
        scores = score_subset_per_fold(
            [0, 1],
            folds=folds,
            X=ds.X,
            Y=ds.Y,
            predict=dummy_predict,
            metric=mean_abs_corr,
        )
        assert scores.shape == (len(folds),)

    def test_single_fold_matches_score_subset(self):
        """With one fold, result matches score_subset."""
        ds = make_dataset(N=50, M=5)
        fold = temporal_fold(50, 0.6)
        indices = [0, 2]

        per_fold = score_subset_per_fold(
            indices,
            folds=[fold],
            X=ds.X,
            Y=ds.Y if ds.Y.ndim == 2 else ds.Y[:, None],
            predict=dummy_predict,
            metric=mean_abs_corr,
        )
        single = score_subset(
            indices,
            X_train=ds.X[fold.train],
            X_validation=ds.X[fold.validation],
            Y_train=ds.Y[fold.train] if ds.Y.ndim == 2 else ds.Y[fold.train, None],
            Y_validation=ds.Y[fold.validation]
            if ds.Y.ndim == 2
            else ds.Y[fold.validation, None],
            predict=dummy_predict,
            metric=mean_abs_corr,
        )
        np.testing.assert_allclose(per_fold[0], single, rtol=1e-10)


# ---------------------------------------------------------------------------
# 8. Weight function tests
# ---------------------------------------------------------------------------


class TestSoftmaxWeight:
    def test_uniform_at_equal_scores(self):
        """Equal fold scores produce uniform weights."""
        fn = softmax_weight(temperature=1.0)
        scores = np.array([0.5, 0.5, 0.5])
        prev = np.zeros(3)
        w = fn(scores, prev, np.full(3, 1.0 / 3))
        np.testing.assert_allclose(w, np.full(3, 1.0 / 3), rtol=1e-10)

    def test_sums_to_one(self):
        """Weights sum to 1."""
        fn = softmax_weight(temperature=2.0)
        scores = np.array([0.1, 0.5, 0.9])
        w = fn(scores, np.zeros(3), np.full(3, 1.0 / 3))
        np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-10)

    def test_low_temperature_concentrates(self):
        """Low temperature concentrates weight on worst fold."""
        fn = softmax_weight(temperature=0.01)
        scores = np.array([0.8, 0.1, 0.5])
        w = fn(scores, np.zeros(3), np.full(3, 1.0 / 3))
        assert w[1] > 0.99  # fold 1 has lowest score

    def test_invalid_temperature(self):
        """Non-positive temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature"):
            softmax_weight(temperature=0.0)
        with pytest.raises(ValueError, match="temperature"):
            softmax_weight(temperature=-1.0)


class TestSoftmaxLossWeight:
    def test_uniform_at_equal_losses(self):
        """Equal losses produce uniform weights."""
        fn = softmax_loss_weight(temperature=1.0)
        losses = np.array([0.5, 0.5, 0.5])
        prev = np.zeros(3)
        w = fn(losses, prev, np.full(3, 1.0 / 3))
        np.testing.assert_allclose(w, np.full(3, 1.0 / 3), rtol=1e-10)

    def test_high_loss_concentrates(self):
        """Low temperature concentrates weight on the highest loss."""
        fn = softmax_loss_weight(temperature=0.01)
        losses = np.array([0.1, 0.9, 0.5])
        w = fn(losses, np.zeros(3), np.full(3, 1.0 / 3))
        assert w[1] > 0.99

    def test_invalid_temperature(self):
        """Non-positive temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature"):
            softmax_loss_weight(temperature=0.0)
        with pytest.raises(ValueError, match="temperature"):
            softmax_loss_weight(temperature=-1.0)


# ---------------------------------------------------------------------------
# 9. Greedy complementary tests
# ---------------------------------------------------------------------------


def _make_split(n: int) -> list[Fold]:
    """Create a simple sliding fold split for tests."""
    ts = max(n // 3, 2)
    vs = max(n // 3, 2)
    folds = sliding_folds(n, train_size=ts, validation_size=vs)
    if not folds:
        # Fallback for very small n: single fold
        mid = n // 2
        folds = [Fold(np.arange(0, max(mid, 1)), np.arange(max(mid, 1), n))]
    return folds


class TestGreedyComplementaryFolds:
    def test_basic_operation(self):
        """Yields at least one step with low threshold."""
        ds = make_dataset()
        steps = list(
            greedy_complementary_folds(
                ds,
                predict=dummy_predict,
                metric=mean_abs_corr,
                split=_make_split,
                max_dim=3,
                threshold=-float("inf"),
            )
        )
        assert len(steps) >= 1
        assert len(steps) <= 3

    def test_threshold_filters(self):
        """High threshold stops selection early."""
        ds = make_dataset()
        steps = list(
            greedy_complementary_folds(
                ds,
                predict=dummy_predict,
                metric=mean_abs_corr,
                split=_make_split,
                max_dim=5,
                threshold=999.0,
            )
        )
        assert len(steps) == 0

    def test_filter_rejects_all(self):
        """Filter that rejects everything yields no steps."""
        ds = make_dataset()

        def reject_all(x: np.ndarray, Y: np.ndarray) -> bool:
            return False

        steps = list(
            greedy_complementary_folds(
                ds,
                predict=dummy_predict,
                metric=mean_abs_corr,
                split=_make_split,
                max_dim=3,
                filter=reject_all,
            )
        )
        assert len(steps) == 0

    def test_max_dim_validation(self):
        """max_dim > N raises ValueError."""
        ds = make_dataset(M=3)
        with pytest.raises(ValueError, match="max_dim"):
            list(
                greedy_complementary_folds(
                    ds,
                    predict=dummy_predict,
                    metric=mean_abs_corr,
                    split=_make_split,
                    max_dim=10,
                )
            )

    def test_empty_split_raises(self):
        """Split producing no folds raises ValueError."""
        ds = make_dataset()

        def empty_split(n: int) -> list[Fold]:
            return []

        with pytest.raises(ValueError, match="no folds"):
            list(
                greedy_complementary_folds(
                    ds,
                    predict=dummy_predict,
                    metric=mean_abs_corr,
                    split=empty_split,
                    max_dim=3,
                )
            )

    def test_no_duplicate_indices(self):
        """Never selects the same variable twice."""
        ds = make_dataset()
        steps = list(
            greedy_complementary_folds(
                ds,
                predict=dummy_predict,
                metric=mean_abs_corr,
                split=_make_split,
                max_dim=5,
                threshold=-float("inf"),
            )
        )
        indices = [s.index for s in steps]
        assert len(indices) == len(set(indices))

    def test_single_fold_equals_greedy(self):
        """With a single fold, complementary matches greedy."""
        ds = make_dataset(N=50, M=4, seed=99)
        fold = temporal_fold(50, 0.6)

        from edmkit.search.dataset import Subset

        train = Subset(ds, fold.train)
        validation = Subset(ds, fold.validation)

        greedy_steps = list(
            greedy(
                train,
                validation,
                predict=colsum_predict,
                metric=neg_mae,
                max_dim=3,
                threshold=-float("inf"),
            )
        )

        def single_fold_split(n: int) -> list[Fold]:
            return [fold]

        comp_steps = list(
            greedy_complementary_folds(
                ds,
                predict=colsum_predict,
                metric=neg_mae,
                split=single_fold_split,
                max_dim=3,
                threshold=-float("inf"),
            )
        )

        assert len(greedy_steps) == len(comp_steps)
        for g, c in zip(greedy_steps, comp_steps):
            assert g.index == c.index
            np.testing.assert_allclose(g.score, c.score, rtol=1e-10)

    def test_softmax_weight_func(self):
        """Works with explicit softmax_weight strategy."""
        ds = make_dataset()
        steps = list(
            greedy_complementary_folds(
                ds,
                predict=dummy_predict,
                metric=mean_abs_corr,
                split=_make_split,
                weight=softmax_weight(temperature=0.1),
                max_dim=3,
                threshold=-float("inf"),
            )
        )
        assert len(steps) >= 1

class TestGreedyComplementaryTimepoints:
    def test_basic_operation(self):
        """Yields at least one step with low threshold."""
        ds = make_dataset()
        steps = list(
            greedy_complementary_timepoints(
                ds,
                predict=dummy_predict,
                metric=mean_abs_corr,
                split=_make_split,
                max_dim=3,
                threshold=-float("inf"),
            )
        )
        assert len(steps) >= 1
        assert len(steps) <= 3

    def test_threshold_filters(self):
        """High threshold stops selection early."""
        ds = make_dataset()
        steps = list(
            greedy_complementary_timepoints(
                ds,
                predict=dummy_predict,
                metric=mean_abs_corr,
                split=_make_split,
                max_dim=5,
                threshold=999.0,
            )
        )
        assert len(steps) == 0

    def test_filter_rejects_all(self):
        """Filter that rejects everything yields no steps."""
        ds = make_dataset()

        def reject_all(x: np.ndarray, Y: np.ndarray) -> bool:
            return False

        steps = list(
            greedy_complementary_timepoints(
                ds,
                predict=dummy_predict,
                metric=mean_abs_corr,
                split=_make_split,
                max_dim=3,
                filter=reject_all,
                threshold=-float("inf"),
            )
        )
        assert len(steps) == 0

    def test_max_dim_validation(self):
        """max_dim > M raises ValueError."""
        ds = make_dataset(M=3)
        with pytest.raises(ValueError, match="max_dim"):
            list(
                greedy_complementary_timepoints(
                    ds,
                    predict=dummy_predict,
                    metric=mean_abs_corr,
                    split=_make_split,
                    max_dim=10,
                )
            )

    def test_empty_split_raises(self):
        """Split producing no folds raises ValueError."""
        ds = make_dataset()

        def empty_split(n: int) -> list[Fold]:
            return []

        with pytest.raises(ValueError, match="no folds"):
            list(
                greedy_complementary_timepoints(
                    ds,
                    predict=dummy_predict,
                    metric=mean_abs_corr,
                    split=empty_split,
                    max_dim=3,
                )
            )

    def test_no_duplicate_indices(self):
        """Never selects the same variable twice."""
        ds = make_dataset()
        steps = list(
            greedy_complementary_timepoints(
                ds,
                predict=dummy_predict,
                metric=mean_abs_corr,
                split=_make_split,
                max_dim=5,
                threshold=-float("inf"),
            )
        )
        indices = [s.index for s in steps]
        assert len(indices) == len(set(indices))

    def test_explicit_loss_weight(self):
        """Works with explicit timepoint weight strategy."""
        ds = make_dataset()
        steps = list(
            greedy_complementary_timepoints(
                ds,
                predict=colsum_predict,
                metric=neg_mae,
                split=_make_split,
                weight=softmax_loss_weight(temperature=0.1),
                max_dim=3,
                threshold=-float("inf"),
            )
        )
        assert len(steps) >= 1


class TestMeanAbsErrorPerSample:
    def test_returns_per_sample_loss(self):
        """Computes one scalar loss per sample."""
        predictions = np.array([[1.0, 3.0], [2.0, 8.0]])
        observations = np.array([[2.0, 1.0], [5.0, 2.0]])
        losses = mean_abs_error_per_sample(predictions, observations)
        np.testing.assert_allclose(losses, np.array([1.5, 4.5]))

    def test_shape_mismatch_raises(self):
        """Mismatched shapes raise ValueError."""
        with pytest.raises(ValueError, match="same shape"):
            mean_abs_error_per_sample(np.ones((3, 1)), np.ones((4, 1)))


class TestMeanSquaredErrorPerSample:
    def test_returns_per_sample_loss(self):
        """Computes one scalar squared loss per sample."""
        predictions = np.array([[1.0, 3.0], [2.0, 8.0]])
        observations = np.array([[2.0, 1.0], [5.0, 2.0]])
        losses = mean_squared_error_per_sample(predictions, observations)
        np.testing.assert_allclose(losses, np.array([2.5, 22.5]))

    def test_shape_mismatch_raises(self):
        """Mismatched shapes raise ValueError."""
        with pytest.raises(ValueError, match="same shape"):
            mean_squared_error_per_sample(np.ones((3, 1)), np.ones((4, 1)))


class TestMeanNegativeCorrelationContributionPerSample:
    def test_returns_negative_contributions(self):
        """Perfect positive correlation yields equal negative contributions."""
        predictions = np.array([[1.0], [2.0], [3.0]])
        observations = np.array([[1.0], [2.0], [3.0]])
        contributions = mean_negative_correlation_contribution_per_sample(
            predictions, observations
        )
        np.testing.assert_allclose(
            contributions,
            np.array([-0.5, 0.0, -0.5]),
            atol=1e-12,
        )

    def test_constant_input_returns_zero(self):
        """Degenerate dimensions contribute zero."""
        predictions = np.array([[1.0], [1.0], [1.0]])
        observations = np.array([[1.0], [2.0], [3.0]])
        contributions = mean_negative_correlation_contribution_per_sample(
            predictions, observations
        )
        np.testing.assert_allclose(contributions, np.zeros(3))

    def test_shape_mismatch_raises(self):
        """Mismatched shapes raise ValueError."""
        with pytest.raises(ValueError, match="same shape"):
            mean_negative_correlation_contribution_per_sample(
                np.ones((3, 1)),
                np.ones((4, 1)),
            )
