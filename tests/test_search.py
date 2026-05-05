"""Tests for the higher-order ``edmkit.search`` API.

Three orthogonal concepts are exercised independently and together:

* **Strategy** — :func:`greedy`, :func:`beam`
* **Evaluation** — :func:`holdout`, :func:`folds`, :func:`loo`,
  :func:`weighted_folds`, :func:`weighted_timepoints`
* **Metric** — borrowed from ``edmkit.metrics``
"""
from __future__ import annotations

import math
import pickle
from functools import partial

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.spatial.distance import cdist

from edmkit.metrics import mae
from edmkit.simplex_projection import simplex_projection
from edmkit.search import (
    Dataset,
    Selection,
    Step,
    beam,
    collect,
    folds,
    greedy,
    holdout,
    loo,
    mean_abs_error_per_sample,
    mean_negative_correlation_contribution_per_sample,
    mean_squared_error_per_sample,
    softmax_loss_weight,
    softmax_weight,
    weighted_folds,
    weighted_timepoints,
)
from edmkit.splits import Fold, sliding_folds

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def make_dataset(
    N: int = 50, M: int = 5, D: int = 1, *, seed: int = 42
) -> Dataset:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, M))
    Y = rng.standard_normal((N, D))
    return Dataset(X=X, Y=Y)


def dummy_predict(
    X: np.ndarray, Y: np.ndarray, Q: np.ndarray, *, mask: np.ndarray | None = None
) -> np.ndarray:
    """1-NN predictor for deterministic tests."""
    del mask
    if X.ndim == 2:
        dists = cdist(Q, X)
        nearest = np.argmin(dists, axis=1)
        return Y[nearest]
    if X.ndim != 3 or Y.ndim != 3 or Q.ndim != 3:
        raise ValueError(f"expected all inputs to be 2D or all to be 3D, got {X.ndim}, {Y.ndim}, {Q.ndim}")
    return np.stack(
        [dummy_predict(X[b], Y[b], Q[b]) for b in range(X.shape[0])],
        axis=0,
    )


def colsum_predict(
    X: np.ndarray, Y: np.ndarray, Q: np.ndarray, *, mask: np.ndarray | None = None
) -> np.ndarray:
    """Cheap analytic predictor: column-sum, broadcast across Y dims."""
    del X, mask
    if Q.ndim == 2 and Y.ndim == 2:
        return Q.sum(axis=1, keepdims=True).repeat(Y.shape[1], axis=1)
    if Q.ndim != 3 or Y.ndim != 3:
        raise ValueError(f"expected Q and Y to be 2D or 3D together, got {Q.ndim} and {Y.ndim}")
    return Q.sum(axis=2, keepdims=True).repeat(Y.shape[2], axis=2)


def mean_abs_corr(predictions: np.ndarray, observations: np.ndarray) -> np.ndarray:
    M = predictions.shape[1]
    total = 0.0
    for m in range(M):
        p = predictions[:, m]
        o = observations[:, m]
        if p.std() < 1e-12 or o.std() < 1e-12:
            continue
        total += abs(float(np.corrcoef(p, o)[0, 1]))
    return np.asarray(total / max(M, 1))


def neg_mae(predictions: np.ndarray, observations: np.ndarray) -> np.ndarray:
    return -mae(predictions, observations)


neg_mae.__name__ = "neg_mae"


def make_holdout(ds: Dataset, *, predict=colsum_predict, metric=neg_mae):
    return holdout(
        X_train=ds.X, X_val=ds.X, Y_train=ds.Y, Y_val=ds.Y,
        predict=predict, metric=metric,
    )


def make_folds_split(n: int) -> list[Fold]:
    ts = max(n // 3, 2)
    vs = max(n // 3, 2)
    split = sliding_folds(n, train_size=ts, validation_size=vs)
    if not split:
        mid = max(n // 2, 1)
        split = [Fold(np.arange(0, mid), np.arange(mid, n))]
    return split


# ---------------------------------------------------------------------------
# Hypothesis strategy for (Dataset, max_dim)
# ---------------------------------------------------------------------------

elems = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)


@st.composite
def search_inputs_impl(draw, *, min_m=2, max_m=8, min_n=8, max_n=30):
    M = draw(st.integers(min_value=min_m, max_value=max_m))
    N = draw(st.integers(min_value=min_n, max_value=max_n))
    max_dim = draw(st.integers(min_value=1, max_value=M))
    X = draw(arrays(np.float64, (N, M), elements=elems))
    Y = draw(arrays(np.float64, (N, 1), elements=elems))
    return Dataset(X=X, Y=Y), max_dim


def search_inputs(*, min_m=2, max_m=8, min_n=8, max_n=30):
    return search_inputs_impl(min_m=min_m, max_m=max_m, min_n=min_n, max_n=max_n)  # ty: ignore[missing-argument]


# ---------------------------------------------------------------------------
# Driver helpers — strategy × evaluation product
# ---------------------------------------------------------------------------


def make_evaluation(kind: str, ds: Dataset):
    if kind == "holdout":
        return make_holdout(ds)
    split = make_folds_split(ds.X.shape[0])
    if kind == "folds":
        return folds(
            X=ds.X, Y=ds.Y, folds=split, predict=colsum_predict, metric=neg_mae,
        )
    if kind == "wfolds":
        return weighted_folds(
            X=ds.X, Y=ds.Y, folds=split, predict=colsum_predict,
            metric=neg_mae, weight_update=softmax_weight(temperature=1.0),
        )
    if kind == "wtime":
        return weighted_timepoints(
            X=ds.X, Y=ds.Y, folds=split, predict=colsum_predict,
            metric=neg_mae, loss=mean_abs_error_per_sample,
            weight_update=softmax_loss_weight(temperature=1.0),
        )
    raise ValueError(kind)


def run_search(algo: str, ds: Dataset, max_dim: int, evaluation) -> list[Step]:
    if algo == "greedy":
        return list(evaluation(greedy, max_dim=max_dim, threshold=-float("inf")))
    if algo == "beam":
        return list(
            evaluation(
                partial(beam, beam_width=2),
                max_dim=max_dim,
                threshold=-float("inf"),
            )
        )
    raise ValueError(algo)


_ALGOS = ["greedy", "beam"]
_KINDS = ["holdout", "folds", "wfolds", "wtime"]


# ---------------------------------------------------------------------------
# 1. Structural invariants — full strategy × evaluation product
# ---------------------------------------------------------------------------


class TestStructuralInvariants:
    @given(data=search_inputs())
    @pytest.mark.parametrize("algo", _ALGOS)
    @pytest.mark.parametrize("kind", _KINDS)
    def test_index_is_last_selected(self, algo, kind, data):
        ds, max_dim = data
        for step in run_search(algo, ds, max_dim, make_evaluation(kind, ds)):
            assert step.index == step.selected[-1]

    @given(data=search_inputs())
    @pytest.mark.parametrize("algo", _ALGOS)
    @pytest.mark.parametrize("kind", _KINDS)
    def test_no_duplicate_indices(self, algo, kind, data):
        ds, max_dim = data
        for step in run_search(algo, ds, max_dim, make_evaluation(kind, ds)):
            assert len(step.selected) == len(set(step.selected))

    @given(data=search_inputs())
    @pytest.mark.parametrize("algo", _ALGOS)
    @pytest.mark.parametrize("kind", _KINDS)
    def test_indices_in_range(self, algo, kind, data):
        ds, max_dim = data
        M = ds.X.shape[1]
        for step in run_search(algo, ds, max_dim, make_evaluation(kind, ds)):
            assert all(0 <= idx < M for idx in step.selected)

    @given(data=search_inputs())
    @pytest.mark.parametrize("algo", _ALGOS)
    @pytest.mark.parametrize("kind", _KINDS)
    def test_selected_within_max_dim(self, algo, kind, data):
        ds, max_dim = data
        for step in run_search(algo, ds, max_dim, make_evaluation(kind, ds)):
            assert len(step.selected) <= max_dim

    @given(data=search_inputs())
    @pytest.mark.parametrize("algo", _ALGOS)
    @pytest.mark.parametrize("kind", _KINDS)
    def test_score_is_finite(self, algo, kind, data):
        ds, max_dim = data
        for step in run_search(algo, ds, max_dim, make_evaluation(kind, ds)):
            assert math.isfinite(step.score)

    @given(data=search_inputs())
    @pytest.mark.parametrize("algo", _ALGOS)
    @pytest.mark.parametrize("kind", _KINDS)
    def test_selected_grows_by_one(self, algo, kind, data):
        ds, max_dim = data
        prev = 0
        for step in run_search(algo, ds, max_dim, make_evaluation(kind, ds)):
            assert len(step.selected) == prev + 1
            prev = len(step.selected)


# ---------------------------------------------------------------------------
# 2. greedy ≡ beam(width=1)
# ---------------------------------------------------------------------------


class TestBeamGreedyEquivalence:
    @given(data=search_inputs())
    @pytest.mark.parametrize("kind", _KINDS)
    def test_beam_width_1_equals_greedy(self, kind, data):
        ds, max_dim = data
        ev = make_evaluation(kind, ds)
        g_steps = list(ev(greedy, max_dim=max_dim, threshold=-float("inf")))
        b_steps = list(
            ev(partial(beam, beam_width=1), max_dim=max_dim, threshold=-float("inf"))
        )
        assert len(g_steps) == len(b_steps)
        for g, b in zip(g_steps, b_steps):
            assert g.index == b.index
            np.testing.assert_allclose(g.score, b.score, rtol=1e-10)
            assert g.selected == b.selected


# ---------------------------------------------------------------------------
# 3. greedy(holdout) reference equivalence
# ---------------------------------------------------------------------------


def _reference_greedy_holdout(
    ds: Dataset, max_dim: int, *, predict, metric, threshold: float
) -> list[Step]:
    """Naive max-by-score reference, untainted by the production loop."""
    M = ds.X.shape[1]
    selected: list[int] = []
    available = list(range(M))
    out: list[Step] = []
    Y2d = ds.Y if ds.Y.ndim == 2 else ds.Y[:, None]
    for _ in range(max_dim):
        best_idx, best_score = -1, float("-inf")
        for c in available:
            p = predict(ds.X[:, selected + [c]], Y2d, ds.X[:, selected + [c]])
            if p.ndim == 1:
                p = p[:, None]
            s = float(metric(p, Y2d))
            if s < threshold:
                continue
            if s > best_score:
                best_idx, best_score = c, s
        if best_idx < 0:
            return out
        selected.append(best_idx)
        available.remove(best_idx)
        out.append(Step(index=best_idx, score=best_score, selected=tuple(selected)))
    return out


class TestHoldoutReference:
    def test_greedy_matches_naive_reference(self):
        ds = make_dataset(N=50, M=5, seed=7)
        reference = _reference_greedy_holdout(
            ds, max_dim=4, predict=colsum_predict, metric=neg_mae,
            threshold=-float("inf"),
        )
        produced = list(
            make_holdout(ds)(greedy, max_dim=4, threshold=-float("inf"))
        )
        assert len(produced) == len(reference)
        for r, p in zip(reference, produced):
            assert r.index == p.index
            np.testing.assert_allclose(r.score, p.score, rtol=1e-10)
            assert r.selected == p.selected


# ---------------------------------------------------------------------------
# 4. Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_max_dim_exceeds_n_candidates(self):
        ds = make_dataset(M=3)
        with pytest.raises(ValueError, match="max_dim"):
            list(make_holdout(ds)(greedy, max_dim=10))

    def test_beam_width_below_1(self):
        ds = make_dataset()
        with pytest.raises(ValueError, match="beam_width"):
            list(make_holdout(ds)(partial(beam, beam_width=0), max_dim=3))

    def test_softmax_weight_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            softmax_weight(temperature=0.0)
        with pytest.raises(ValueError, match="temperature"):
            softmax_loss_weight(temperature=-1.0)

    def test_loo_negative_tau(self):
        with pytest.raises(ValueError, match="tau"):
            loo(X=np.zeros((5, 2)), Y=np.zeros(5), metric=neg_mae, tau=-1)

    def test_folds_empty(self):
        with pytest.raises(ValueError, match="folds"):
            folds(
                X=np.zeros((5, 2)), Y=np.zeros(5), folds=[],
                predict=colsum_predict, metric=neg_mae,
            )


# ---------------------------------------------------------------------------
# 5. Threshold and filter behaviour
# ---------------------------------------------------------------------------


class TestThresholdAndFilter:
    def test_threshold_filters_to_zero(self):
        ds = make_dataset()
        ev = make_holdout(ds, predict=dummy_predict, metric=mean_abs_corr)
        steps = list(ev(greedy, max_dim=5, threshold=999.0))
        assert steps == []

    def test_filter_rejects_all(self):
        ds = make_dataset()
        ev = make_holdout(ds, predict=dummy_predict, metric=mean_abs_corr)
        reject_all = lambda i: False  # noqa: E731
        reject_all.__name__ = "reject_all"
        steps = list(
            ev(greedy, max_dim=3, threshold=-float("inf"), filter=reject_all)
        )
        assert steps == []

    def test_filter_blocks_specific_indices(self):
        ds = make_dataset(M=4)
        ev = make_holdout(ds, predict=dummy_predict, metric=mean_abs_corr)
        blocked = {0, 1}
        not_blocked = lambda i: i not in blocked  # noqa: E731
        not_blocked.__name__ = "not_blocked"
        steps = list(
            ev(
                partial(beam, beam_width=2),
                max_dim=2,
                threshold=-float("inf"),
                filter=not_blocked,
            )
        )
        for step in steps:
            for idx in step.selected:
                assert idx not in blocked

    @given(data=search_inputs(), t_offset=st.floats(0.01, 100.0))
    def test_threshold_monotonicity(self, data, t_offset):
        ds, max_dim = data
        ev = make_holdout(ds)
        steps_lo = list(ev(greedy, max_dim=max_dim, threshold=-float("inf")))
        steps_hi = list(ev(greedy, max_dim=max_dim, threshold=t_offset))
        assert len(steps_lo) >= len(steps_hi)


# ---------------------------------------------------------------------------
# 6. Adaptive evaluation: initial-step semantics & state shape
# ---------------------------------------------------------------------------


class _CapturingStrategy:
    """Runs greedy but captures the first-step score of candidate ``c=0``.

    Used to verify that adaptive evaluations agree with static ones on the
    first step when initial state is zeros + softmax → uniform weights.
    """

    def __init__(self):
        self.first_step_scores: dict[int, float] = {}

    def __call__(
        self, evaluation, *, max_dim, threshold=0.0, filter=None,
    ):
        del max_dim, threshold, filter
        path = evaluation.initial_path()
        candidates = np.arange(evaluation.n_candidates, dtype=np.intp)
        for scored in evaluation.evaluate_frontier([path], [candidates]):
            self.first_step_scores[scored.candidate] = scored.score
        return iter(())


class TestWeightedFolds:
    def test_first_step_uniform_matches_folds_mean(self):
        """zeros + softmax_weight ⇒ uniform first step == folds mean."""
        ds = make_dataset(N=60, M=4, seed=11)
        split = make_folds_split(60)
        ev_w = weighted_folds(
            X=ds.X, Y=ds.Y, folds=split, predict=colsum_predict,
            metric=neg_mae, weight_update=softmax_weight(temperature=1.0),
        )
        ev_f = folds(
            X=ds.X, Y=ds.Y, folds=split, predict=colsum_predict, metric=neg_mae,
        )
        cap_w, cap_f = _CapturingStrategy(), _CapturingStrategy()
        list(ev_w(cap_w, max_dim=1, threshold=-float("inf")))
        list(ev_f(cap_f, max_dim=1, threshold=-float("inf")))
        for c in range(ds.X.shape[1]):
            np.testing.assert_allclose(
                cap_w.first_step_scores[c],
                cap_f.first_step_scores[c],
                rtol=1e-10,
            )

    def test_state_evolves_to_fold_scores(self):
        ds = make_dataset(N=60, M=4, seed=13)
        split = make_folds_split(60)
        ev = weighted_folds(
            X=ds.X, Y=ds.Y, folds=split, predict=colsum_predict,
            metric=neg_mae, weight_update=softmax_weight(temperature=1.0),
        )
        # Drive through greedy and make sure two real steps run to completion.
        steps = list(ev(greedy, max_dim=2, threshold=-float("inf")))
        assert len(steps) == 2


class TestWeightedTimepoints:
    def test_first_step_uniform_matches_neg_mean_loss(self):
        """zeros + softmax_loss_weight ⇒ first step ∝ -mean(losses)."""
        ds = make_dataset(N=60, M=4, seed=17)
        split = make_folds_split(60)
        ev = weighted_timepoints(
            X=ds.X, Y=ds.Y, folds=split, predict=colsum_predict,
            metric=neg_mae, loss=mean_abs_error_per_sample,
            weight_update=softmax_loss_weight(temperature=1.0),
        )
        cap = _CapturingStrategy()
        list(ev(cap, max_dim=1, threshold=-float("inf")))
        # For each candidate, the uniform-weighted improvement reduces to
        # ``-mean(new_sample_losses)``. Recompute directly to verify.
        for c in range(ds.X.shape[1]):
            losses_parts = []
            for fold in split:
                p = colsum_predict(
                    ds.X[fold.train][:, [c]],
                    ds.Y[fold.train],
                    ds.X[fold.validation][:, [c]],
                )
                losses_parts.append(mean_abs_error_per_sample(p, ds.Y[fold.validation]))
            losses = np.concatenate(losses_parts)
            expected = -float(losses.mean())
            np.testing.assert_allclose(
                cap.first_step_scores[c], expected, rtol=1e-6,
            )


# ---------------------------------------------------------------------------
# 7. LOO evaluation
# ---------------------------------------------------------------------------


class TestLOO:
    def test_runs_with_greedy(self):
        ds = make_dataset(N=80, M=5, seed=29)
        ev = loo(X=ds.X, Y=ds.Y, metric=neg_mae, tau=1)
        steps = list(ev(greedy, max_dim=3, threshold=-float("inf")))
        assert len(steps) == 3

    def test_runs_with_beam(self):
        ds = make_dataset(N=80, M=5, seed=31)
        ev = loo(X=ds.X, Y=ds.Y, metric=neg_mae, tau=1)
        steps = list(
            ev(partial(beam, beam_width=2), max_dim=3, threshold=-float("inf"))
        )
        assert len(steps) == 3


# ---------------------------------------------------------------------------
# 8. Specialized simplex frontier matches generic callback semantics
# ---------------------------------------------------------------------------


class TestSimplexSpecialization:
    def test_holdout_specialization_matches_generic_wrapper(self):
        ds = make_dataset(N=60, M=5, seed=101)

        def wrapped_predict(X, Y, Q, *, mask=None):
            return simplex_projection(X, Y, Q, mask=mask)

        direct = holdout(
            X_train=ds.X,
            X_val=ds.X,
            Y_train=ds.Y,
            Y_val=ds.Y,
            predict=simplex_projection,
            metric=neg_mae,
        )
        wrapped = holdout(
            X_train=ds.X,
            X_val=ds.X,
            Y_train=ds.Y,
            Y_val=ds.Y,
            predict=wrapped_predict,
            metric=neg_mae,
        )

        direct_steps = list(direct(greedy, max_dim=3, threshold=-float("inf")))
        wrapped_steps = list(wrapped(greedy, max_dim=3, threshold=-float("inf")))
        assert [s.index for s in direct_steps] == [s.index for s in wrapped_steps]
        for actual, expected in zip(direct_steps, wrapped_steps, strict=True):
            np.testing.assert_allclose(actual.score, expected.score, rtol=1e-5, atol=1e-5)
            assert actual.selected == expected.selected

    def test_folds_specialization_matches_generic_wrapper(self):
        ds = make_dataset(N=72, M=6, seed=103)
        split = make_folds_split(len(ds.X))

        def wrapped_predict(X, Y, Q, *, mask=None):
            return simplex_projection(X, Y, Q, mask=mask)

        direct = folds(
            X=ds.X,
            Y=ds.Y,
            folds=split,
            predict=simplex_projection,
            metric=neg_mae,
        )
        wrapped = folds(
            X=ds.X,
            Y=ds.Y,
            folds=split,
            predict=wrapped_predict,
            metric=neg_mae,
        )

        direct_steps = list(direct(partial(beam, beam_width=2), max_dim=3, threshold=-float("inf")))
        wrapped_steps = list(wrapped(partial(beam, beam_width=2), max_dim=3, threshold=-float("inf")))
        assert [s.index for s in direct_steps] == [s.index for s in wrapped_steps]
        for actual, expected in zip(direct_steps, wrapped_steps, strict=True):
            np.testing.assert_allclose(actual.score, expected.score, rtol=1e-5, atol=1e-5)
            assert actual.selected == expected.selected

    def test_weighted_timepoints_specialization_matches_generic_wrapper(self):
        ds = make_dataset(N=72, M=5, seed=107)
        split = make_folds_split(len(ds.X))

        def wrapped_predict(X, Y, Q, *, mask=None):
            return simplex_projection(X, Y, Q, mask=mask)

        direct = weighted_timepoints(
            X=ds.X,
            Y=ds.Y,
            folds=split,
            predict=simplex_projection,
            metric=neg_mae,
            loss=mean_abs_error_per_sample,
            weight_update=softmax_loss_weight(temperature=1.0),
        )
        wrapped = weighted_timepoints(
            X=ds.X,
            Y=ds.Y,
            folds=split,
            predict=wrapped_predict,
            metric=neg_mae,
            loss=mean_abs_error_per_sample,
            weight_update=softmax_loss_weight(temperature=1.0),
        )

        direct_steps = list(direct(greedy, max_dim=3, threshold=-float("inf")))
        wrapped_steps = list(wrapped(greedy, max_dim=3, threshold=-float("inf")))
        assert [s.index for s in direct_steps] == [s.index for s in wrapped_steps]
        for actual, expected in zip(direct_steps, wrapped_steps, strict=True):
            np.testing.assert_allclose(actual.score, expected.score, rtol=1e-5, atol=1e-5)
            assert actual.selected == expected.selected


# ---------------------------------------------------------------------------
# 9. Per-sample loss helpers
# ---------------------------------------------------------------------------


class TestMeanAbsErrorPerSample:
    def test_per_sample_mae(self):
        predictions = np.array([[1.0, 3.0], [2.0, 8.0]])
        observations = np.array([[2.0, 1.0], [5.0, 2.0]])
        np.testing.assert_allclose(
            mean_abs_error_per_sample(predictions, observations),
            np.array([1.5, 4.5]),
        )

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="same shape"):
            mean_abs_error_per_sample(np.ones((3, 1)), np.ones((4, 1)))


class TestMeanSquaredErrorPerSample:
    def test_per_sample_mse(self):
        predictions = np.array([[1.0, 3.0], [2.0, 8.0]])
        observations = np.array([[2.0, 1.0], [5.0, 2.0]])
        np.testing.assert_allclose(
            mean_squared_error_per_sample(predictions, observations),
            np.array([2.5, 22.5]),
        )

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="same shape"):
            mean_squared_error_per_sample(np.ones((3, 1)), np.ones((4, 1)))


class TestMeanNegativeCorrelationContributionPerSample:
    def test_perfect_correlation(self):
        predictions = np.array([[1.0], [2.0], [3.0]])
        observations = np.array([[1.0], [2.0], [3.0]])
        contribs = mean_negative_correlation_contribution_per_sample(
            predictions, observations
        )
        np.testing.assert_allclose(contribs, np.array([-0.5, 0.0, -0.5]), atol=1e-12)

    def test_constant_input(self):
        predictions = np.array([[1.0], [1.0], [1.0]])
        observations = np.array([[1.0], [2.0], [3.0]])
        contribs = mean_negative_correlation_contribution_per_sample(
            predictions, observations
        )
        np.testing.assert_allclose(contribs, np.zeros(3))

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="same shape"):
            mean_negative_correlation_contribution_per_sample(
                np.ones((3, 1)), np.ones((4, 1))
            )


# ---------------------------------------------------------------------------
# 10. Softmax weight functions
# ---------------------------------------------------------------------------


class TestSoftmaxWeight:
    def test_zero_state_is_uniform(self):
        fn = softmax_weight(temperature=1.0)
        w = fn(np.zeros(4))
        np.testing.assert_allclose(w, np.full(4, 0.25), rtol=1e-10)

    def test_low_temperature_concentrates_on_low_score(self):
        fn = softmax_weight(temperature=0.01)
        w = fn(np.array([0.8, 0.1, 0.5]))
        assert w[1] > 0.99


class TestSoftmaxLossWeight:
    def test_zero_state_is_uniform(self):
        fn = softmax_loss_weight(temperature=1.0)
        w = fn(np.zeros(4))
        np.testing.assert_allclose(w, np.full(4, 0.25), rtol=1e-10)

    def test_low_temperature_concentrates_on_high_loss(self):
        fn = softmax_loss_weight(temperature=0.01)
        w = fn(np.array([0.1, 0.9, 0.5]))
        assert w[1] > 0.99


# ---------------------------------------------------------------------------
# 11. collect()
# ---------------------------------------------------------------------------


class TestCollect:
    def test_empty(self):
        result = collect(iter([]))
        assert result.indices == []
        assert result.scores == []

    def test_uses_last_selected(self):
        steps = [
            Step(index=2, score=0.5, selected=(2,)),
            Step(index=5, score=0.7, selected=(2, 5)),
            Step(index=1, score=0.8, selected=(2, 5, 1)),
        ]
        sel = collect(steps)
        assert sel.indices == [2, 5, 1]
        assert sel.scores == [0.5, 0.7, 0.8]

    def test_from_greedy(self):
        ds = make_dataset()
        ev = make_holdout(ds, predict=dummy_predict, metric=mean_abs_corr)
        sel = collect(ev(greedy, max_dim=3, threshold=-float("inf")))
        assert isinstance(sel, Selection)
        assert len(sel.indices) == len(sel.scores)


# ---------------------------------------------------------------------------
# 12. Pickleability of strategies with partial kwargs
# ---------------------------------------------------------------------------


class TestStrategyPickle:
    def test_partial_beam_pickleable(self):
        s = partial(beam, beam_width=3)
        s2 = pickle.loads(pickle.dumps(s))
        assert s2.keywords == {"beam_width": 3}

    def test_partial_greedy_with_threshold_pickleable(self):
        s = partial(greedy, threshold=-float("inf"))
        s2 = pickle.loads(pickle.dumps(s))
        assert s2.keywords["threshold"] == -float("inf")
