# AGENTS.md

## Project Overview

`edmkit-search` (namespace `edmkit.search`) is a Python library for variable
selection and causal inference: greedy/beam/annealing variable selection, CCM
convergence testing, AICc-based model comparison, dataset management, and
prediction metrics for EDM workflows. The main external dependency is `edmkit`,
which provides core EDM primitives such as bootstrap CCM and lagged embedding.
Both packages use `uv_build` with `namespace = true` to coexist under the
`edmkit` namespace root.

## Project Structure

Keep module boundaries sharp and dependencies flowing downward:

```text
src/edmkit/search/
  convergence.py  <- CCM convergence tests using aicc + edmkit
  greedy.py       <- greedy variable selection
  beam.py         <- beam search variable selection
  annealing.py    <- simulated annealing variable selection
  common.py       <- shared helpers (prepare_data, score_subset)
  aicc.py         <- linear vs saturation model comparison
  metrics.py      <- scalar and per-dimension prediction metrics
  types.py        <- protocol types (PredictFn, MetricFn, FilterFn) + Step, Selection
  dataset/        <- Dataset, Subset, splits, transforms, DataLoader
  data/           <- dataset-specific loaders and simulators
tests/            <- pytest + hypothesis coverage
e2e/fly.py        <- main integration smoke path against data/fly.csv
experiments/      <- exploratory analysis and generated output, not package API
```

No `__init__.py` in `src/edmkit/` (namespace root). Subpackages keep theirs.

Prefer extending the existing layer where the responsibility already fits
instead of introducing cross-cutting helpers.

## Development Commands

Use `uv` for local work:

- `uv sync --dev` installs runtime and development dependencies from `uv.lock`.
- `uv run pytest` runs the default suite with `-x --tb=short`.
- `uv run pytest tests/test_metrics.py::test_self_correlation_per_dim` runs one
  focused test.
- `uv run pytest --hypothesis-profile=ci` runs the thorough Hypothesis profile.
- `uv run ruff check .` runs linting.
- `uv run ty check src tests` runs static type checks.
- `uv run python e2e/fly.py` runs the end-to-end smoke test.
- `uv build` creates the sdist and wheel in `dist/`.

## Coding Conventions

Follow `CODING.md`. The important defaults are:

- Prefer functional-style code, immutable data flow, and fail-fast validation.
- Inject variable behavior through protocol-typed callables such as `PredictFn`,
  `MetricFn`, and `FilterFn`; do not add stringly-typed mode switches when a
  callable strategy fits.
- Use `functools.partial` for parameterized strategies instead of branching
  inside core logic.
- Keep public APIs fully typed and document public functions with NumPy-style
  docstrings.
- Use `snake_case` for functions, variables, modules, and tests; use
  `PascalCase` for type aliases and container types.
- Preserve standard EDM notation such as `E`, `tau`, and `theta`.
- Treat arrays as `(N, M)` by default; normalize 1D inputs at module boundaries
  and keep shape handling explicit.
- Keep `__init__.py` files focused on re-exports rather than implementation
  logic.

## Testing Expectations

Pytest and Hypothesis are the primary test tools.

- Add tests in `tests/test_*.py`.
- Reuse shared strategies from `tests/strategies.py` where possible.
- `tests/conftest.py` defaults to the fast `dev` profile; use
  `--hypothesis-profile=ci` for deeper checks when behavior changes materially.
- Favor deterministic assertions and cover both successful shape handling and
  validation failures.
- Add regression tests when changing greedy search, metrics, transforms, or
  convergence behavior.

## Change Guidance

- Keep library code in `src/edmkit/search/`; avoid turning `experiments/` code
  into implicit package API.
- Preserve `Dataset` and `Subset` interchangeability and other protocol-based
  seams when refactoring.
- When behavior changes, describe the impact clearly and record the verification
  commands you ran.
- Use short imperative commit subjects, but prefer specific summaries over vague
  messages like `wip`.
