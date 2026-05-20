# edmkit-search

Trajectory construction on an energy landscape, packaged as a library on top
of [`edmkit`](https://github.com/FujishigeTemma/edmkit).

## Install

```bash
pip install edmkit-search
# or
uv add edmkit-search
```

## Overview

The search loop builds a trajectory step by step:

1. From the current frontier (batch of `(state, context)`),
2. expand neighbors via a `Neighborhood`,
3. score children with an `Energy` to obtain `(energies, contexts)`,
4. and pick the next frontier with a `Strategy`.

See `CODING.md` for the design rules. The subpackages mirror the four
abstractions: `energy/`, `neighborhood/`, `state/`, `strategy/`, plus
`dataset/` for the input containers.

## Quick start

```bash
uv sync
PYTHON_GIL=0 uv run python e2e/synthetic.py
```

`e2e/synthetic.py` generates a Lorenz-96 trajectory mixed with noise columns
and uses greedy forward selection to recover the informative subset.

## Tests

```bash
uv run pytest
```
