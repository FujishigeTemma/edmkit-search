# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

**edmkit-search** (namespace `edmkit.search`) is a Python library for
variable selection and causal inference built on Empirical Dynamic Modeling
(EDM). It provides greedy/beam/annealing variable selection, convergent
cross-mapping (CCM) convergence testing via AICc model comparison, and
time-series dataset management.

Core dependency: `edmkit>=0.0.4` (external EDM library providing
`ccm.bootstrap`, `embedding.lagged_embed`, `metrics`, `splits`, `types`,
etc.). Both packages use `uv_build` with `namespace = true` so `edmkit.search`
and `edmkit.ccm` etc. coexist as namespace packages under the `edmkit` root.

## Commands

```bash
# Install dependencies (uses uv, uv_build backend)
uv sync --dev

# Run all tests (pytest with hypothesis, defaults to "dev" profile: 10 examples)
# pyproject.toml applies -x --tb=short
uv run pytest

# Run a single test file or test
uv run pytest tests/test_search.py
uv run pytest tests/test_search.py::TestGreedy

# Hypothesis profiles: dev (default, fast), ci (500 examples), debug (verbose)
uv run pytest --hypothesis-profile=ci

# Lint and type check
uv run ruff check .
uv run ty check src tests

# Build
uv build

# E2E smoke test against data/fly.csv
uv run python e2e/fly.py
```

## Architecture

### Module layers (dependency flows downward)

```
src/edmkit/search/
  convergence.py  ← CCM convergence test (uses aicc + edmkit)
  greedy.py       ← Greedy variable selection
  beam.py         ← Beam search variable selection
  annealing.py    ← Simulated annealing variable selection
  common.py       ← Shared helpers (prepare_data, score_subset, negate)
  aicc.py         ← AICc model comparison (linear vs saturation curve fitting)
  types.py        ← FilterFn protocol + Step, Selection NamedTuples
  dataset/        ← Data containers, transforms, DataLoader
    containers.py ← Dataset, Subset (Subset is a zero-copy view)
    transforms.py ← Transform type alias, zscore_normalize, gaussian_noise, compose
    loader.py     ← DataLoader (mini-batch iterator)
  data/           ← Data loaders (fly.py uses polars, lorenz96.py is a simulator)
```

No `__init__.py` in `src/edmkit/` (namespace root). Subpackages keep theirs.

Keep module boundaries sharp: each module has a single responsibility as shown
above. Treat `experiments/` as analysis code and generated output, not package
API. `e2e/` contains end-to-end smoke tests using real datasets.

### Key patterns

- **Protocol-based DI**: Core behavior is injected via `PredictFunc` (edmkit),
  `MetricFunc` (edmkit), `FilterFn` (search-specific) protocols. Use
  `functools.partial` to bind parameters.
- **Dataset/Subset interchangeability**: `Subset` exposes `.X`/`.Y` properties
  matching `Dataset`, so both work with search functions.
- **Search algorithms**: `greedy`, `beam`, `anneal` are generators yielding
  `Step` objects; `collect()` aggregates into `Selection`.
- **CCM convergence** (`convergence.causation`): Embeds Y (effect), predicts X
  (cause) via bootstrap sampling, then compares linear vs saturation AICc to
  detect convergence.

### Data shapes convention

Arrays are consistently `(N, M)` — N samples, M dimensions. 1D inputs are
auto-promoted to `(N, 1)` at boundaries.

## Coding Style

Detailed in `CODING.md`. Key points:

- Functional style; immutability (no in-place mutation of inputs)
- Domain notation preserved in variable names (`tau`, `E`, `theta`, not
  `time_delay`, `embedding_dimension`)
- Shape variables use uppercase single letters with inline comments:
  `B, N, E = X.shape`
- NumPy-style docstrings with Parameters, Returns, Raises sections
- Keyword-only arguments (`*`) for non-obvious parameters
- Fail-fast validation with expected vs actual values in error messages
- `PascalCase` for type aliases and `NamedTuple` containers; `snake_case` for
  everything else
- 4-space indentation, explicit type annotations

## Testing

Tests use **hypothesis** for property-based testing. Shared strategies live in
`tests/strategies.py` (`arrays_2d`, `matched_arrays`, `reasonable_floats`). The
default hypothesis profile is "dev" (10 examples) configured in
`tests/conftest.py`.

Write new tests as `tests/test_*.py`. Favor deterministic assertions, and cover
both shape handling and validation errors. Preserve the array convention: most
APIs expect `(N, M)`, while 1D targets are promoted to `(N, 1)` at module
boundaries. Add regression tests when changing metrics, dataset transforms, or
greedy search behavior.

## Commit & Review Guidelines

Short imperative subjects (e.g. `tighten greedy search validation`,
`fix metric shape check`). Avoid vague subjects like `wip`. For reviewable
changes, include a short description of behavioral impact, list the verification
commands you ran, and attach plots or output snippets when `e2e/` or
`experiments/` results change.
