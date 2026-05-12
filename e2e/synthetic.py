import os
import sysconfig
import warnings

# usearch.compiled (transitively imported below) re-enables the GIL on the
# free-threaded build unless PYTHON_GIL=0 is set at interpreter startup,
# which neutralizes the thread pool below.
if sysconfig.get_config_var("Py_GIL_DISABLED") and os.environ.get("PYTHON_GIL") != "0":
    warnings.warn(
        "Run with PYTHON_GIL=0 to keep the GIL disabled; otherwise the thread pool below will not scale.",
        stacklevel=2,
    )

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from edmkit.embedding import scan, select
from edmkit.metrics import mean_rho as _mean_rho
from edmkit.simplex_projection import simplex_projection
from edmkit.splits import Fold, temporal_fold

from edmkit.search import energy, neighborhood, state, strategy
from edmkit.search.dataset import Dataset, Subset
from edmkit.search.energy import Contexts, Energies, Energy, Plan
from edmkit.search.state import States


def mean_rho(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    return 1.0 - _mean_rho(predicted.reshape(observed.shape), observed)


def parallel(initial: Contexts, plan: Plan, pool: ThreadPoolExecutor) -> Energy:
    def E(states: States, contexts: Contexts) -> tuple[Energies, Contexts]:
        futures = [pool.submit(job) for job in plan(states, contexts)]
        n = states.shape[0]
        energies = np.empty(n, dtype=np.float64)
        new_contexts = np.empty((n, initial.shape[1]), dtype=np.float64)
        for f in futures:
            s, e, c = f.result()
            energies[s] = e
            new_contexts[s] = c
        return energies, new_contexts

    return E


def lorenz96(
    *,
    K: int,
    T: int,
    F: float = 8.0,
    dt: float = 0.05,
    rng: np.random.Generator,
) -> np.ndarray:
    x = F + 0.01 * rng.standard_normal(K)

    def deriv(x: np.ndarray) -> np.ndarray:
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

    traj = np.empty((T, K), dtype=np.float64)
    for t in range(T):
        k1 = deriv(x)
        k2 = deriv(x + 0.5 * dt * k1)
        k3 = deriv(x + 0.5 * dt * k2)
        k4 = deriv(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj[t] = x
    return traj


def main() -> None:
    rng = np.random.default_rng(0)

    K_signal = 6
    K_noise = 18
    T = 2000
    horizon = 1

    signal = lorenz96(K=K_signal, T=T + horizon, rng=rng)
    noise = np.empty((T + horizon, K_noise), dtype=np.float64)
    noise[0] = rng.standard_normal(K_noise)
    phi = rng.uniform(0.2, 0.8, size=K_noise)
    for t in range(1, T + horizon):
        noise[t] = phi * noise[t - 1] + rng.standard_normal(K_noise)

    full = np.concatenate([signal, noise], axis=1)
    order = rng.permutation(full.shape[1])
    X_full = full[:T, order]
    informative_idx = {int(np.where(order == k)[0][0]) for k in range(K_signal)}

    # Predict the next-step value of L96 site 0 (original signal column 0).
    Y_full = signal[horizon : T + horizon, 0:1]

    data = Dataset(X=X_full, Y=Y_full)
    print(f"Dataset: X{data.X.shape}, Y{data.Y.shape}")
    print(f"Informative columns (after shuffle): {sorted(informative_idx)}")

    outer = temporal_fold(len(data), train_ratio=0.8)
    train = Subset(data, outer.train)
    validation = Subset(data, outer.validation)
    print(f"Outer fold - train: {len(train)}, validation: {len(validation)}")

    E_range = list(range(1, 10 + 1))
    tau_range = list(range(1, 5 + 1))
    threshold = 0.1

    # filter variables if embedding of the variable doesn't predict the target manifold enough with optimal embedding parameters (E and tau)
    def best_score(i: int) -> float:
        return float(
            select(
                scan(
                    train.X[:, i],
                    train.Y,
                    E=E_range,
                    tau=tau_range,
                    predict=simplex_projection,
                    metric=_mean_rho,
                ),
                E=E_range,
                tau=tau_range,
            )[2]
        )

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        best_scores = np.fromiter(
            pool.map(best_score, range(data.X.shape[1])),
            dtype=np.float64,
            count=data.X.shape[1],
        )

        mask = best_scores >= threshold
        print(f"Retaining {int(mask.sum())} out of {data.X.shape[1]} variables after filtering with threshold={threshold}")

        kept = np.flatnonzero(mask).tolist()
        informative_idx = {new_idx for new_idx, old_idx in enumerate(kept) if old_idx in informative_idx}
        data = Dataset(X=data.X[:, mask], Y=data.Y)
        train = Subset(data, outer.train)
        validation = Subset(data, outer.validation)
        print(f"Informative columns after filter: {sorted(informative_idx)}")

        inner = temporal_fold(len(train), train_ratio=0.75)
        print(f"Inner fold - train: {len(inner.train)}, validation: {len(inner.validation)}")

        initial_ctx, plan = energy.holdout(
            data=train,
            fold=Fold(train=inner.train, validation=inner.validation),
            predict=simplex_projection,
            metric=mean_rho,
            batch_size=64,
        )
        E = parallel(initial_ctx, plan, pool)
        N = neighborhood.forward(data.X.shape[1])
        step = strategy.greedy(E, N)
        initial = strategy.Frontier(
            states=state.initial(),
            contexts=initial_ctx,
            energies=np.array([float("inf")], dtype=np.float64),
        )

        max_steps = K_signal + 2
        trace = list(strategy.run(initial, step, max_steps=max_steps, rng=rng))

    if not trace:
        print("No variables selected.")
        return

    selected = [int(i) for i in trace[-1].states[0]]
    scores = [float(frontier.energies[0]) for frontier in trace]
    print(f"\nSelected {len(selected)} variables (greedy):")
    for i, (idx, score) in enumerate(zip(selected, scores), 1):
        marker = "*" if idx in informative_idx else " "
        print(f"  {marker} dim {i}: idx={idx:3d}, score={score:.4f}")

    recovered = sum(1 for idx in selected if idx in informative_idx)
    print(f"\nRecovered {recovered}/{K_signal} informative variables in the first {len(selected)} selections.")

    predictions = simplex_projection(train.X[:, selected], train.Y, validation.X[:, selected])
    score = float(mean_rho(predictions, validation.Y))
    print(f"Held-out validation score (1 - mean_rho): {score:.4f}")


if __name__ == "__main__":
    main()
