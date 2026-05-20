---
title: The Search Loop
description: The one loop and four abstractions that edmkit-search is built around.
---

edmkit-search is a small library wrapped around a single loop: build a trajectory of candidate states by repeatedly expanding the current frontier, scoring the children, and keeping the best.

## The Loop

The search carries a single piece of state: a **Frontier** — a batch of candidate states paired with their scores.

```python
@dataclass(frozen=True)
class Frontier:
    states: States      # (N, d) — N candidates, each d indices into the dataset
    contexts: Contexts  # (N, K) — energy-side data threaded between steps
    energies: Energies  # (N,)  — score per candidate (lower is better)
```

`run` drives the loop. Starting from an `initial` frontier, it applies a `step` up to `max_steps` times. Each step **expands** the frontier into children, **scores** them, and **keeps** the survivors — that becomes the next frontier. After every step, `run` yields the *single* lowest-energy survivor as a one-row frontier; the *full* post-step frontier is still threaded into the next iteration internally.

```mermaid
flowchart TD
    I["initial Frontier<br/>states (1, 0) — empty<br/>energies [∞]"]
    F(["Frontier<br/>(N, d)"])
    N["Neighborhood — expand"]
    E["Energy — score"]
    Sel["select<br/>argsort + keep width best"]
    Fout(["Frontier'<br/>(W, d+1)"])
    T[("trace<br/>one row per step")]
    I --> F
    F -->|parent states| N
    N -->|"M children + parents_idx (M,)"| E
    E -->|"energies (M,), contexts (M, K)"| Sel
    Sel --> Fout
    Fout -.->|fed to next iteration| F
    Fout -->|argmin row<br/>yielded by run| T
```

The three boxes inside the cycle are what a `Strategy` (`greedy`, `beam`) composes — the strategy **is** the step `(Frontier, rng) -> Frontier'`. The fourth piece, `Neighborhood`, hands children to `Energy` with a `parents_idx` back-pointer so the strategy can lift parent-side data onto the children (`E(children, frontier.contexts[parents_idx])`); the `Neighborhood` itself never sees the contexts.

For the standard [`forward`](/edmkit-search/reference/neighborhood/) neighborhood, each iteration grows `d` by 1 — so starting from `(1, 0)`, **`trace[j].states[0]` is the selected index set of length `j + 1`**. The iterator stops early when a step returns an empty frontier (the neighborhood ran out of children) or after `max_steps` iterations, whichever comes first.

## The Four Pieces

| Piece | Type signature |
| ----- | -------------- |
| **State** | `States = NDArray[int64]` of shape `(N, d)` |
| **Neighborhood** | `(parents, rng) -> (children, parents_idx)` |
| **Energy** | `(states, contexts) -> (energies, contexts)` — also has a deferred-job `Plan` form for parallel scoring, see [Energy](/edmkit-search/concepts/energy/) |
| **Strategy** | `Step = (Frontier, rng) -> Frontier'` |

There is no base class. The four abstractions are plain callable protocols (and one frozen `dataclass`), so a custom expansion rule, scorer, or selection policy drops in anywhere without subclassing — see the "Writing Your Own" sections of [Neighborhood](/edmkit-search/concepts/neighborhood/) and [Energy](/edmkit-search/concepts/energy/).

## Anatomy of One Step

For reference, every built-in strategy reduces to this body — the rest of the library is plumbing around it:

```python
def step(frontier: Frontier, rng: np.random.Generator) -> Frontier:
    children, parents_idx = N(frontier.states, rng)              # (M, d+1), (M,)
    energies, contexts = E(children, frontier.contexts[parents_idx])  # (M,), (M, K)
    order = np.argsort(energies, kind="stable")[:width]          # keep best `width`
    return Frontier(children[order], contexts[order], energies[order])
```

The runtime cost lives entirely inside `E(...)`. Everything around it is index gymnastics.

## Energies Are Minimized

Strategies pick the rows with the *lowest* energy. When the natural quantity is a goodness score (e.g. Pearson correlation), wrap it as `1 - score` before handing it to an energy.

## Reproducibility

`run` accepts an `np.random.Generator` and threads it through every `Step` invocation. Neighborhoods may consume it for randomized expansion order (see [`forward`](/edmkit-search/reference/neighborhood/)); energies typically do not. A single seed reproduces the entire trajectory.

## What's Next

- [Energy](/edmkit-search/concepts/energy/) — the `Plan` / `Energy` split, the three built-in scorers, and how to write your own.
- [Neighborhood](/edmkit-search/concepts/neighborhood/) — `forward` and how to write other expansion rules.
- [Strategy](/edmkit-search/concepts/strategy/) — `greedy`, `beam`, and the initial-frontier conventions.
- [`e2e/synthetic.py`](https://github.com/FujishigeTemma/edmkit-search/blob/main/e2e/synthetic.py) — the full reference pipeline (filter → fold → search → score).
