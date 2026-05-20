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
from edmkit.splits import temporal_fold

from edmkit.search import energy, neighborhood, state, strategy
from edmkit.search.dataset import Dataset, Subset


def mean_rho(predictions: np.ndarray, observations: np.ndarray) -> np.ndarray:
    return 1.0 - _mean_rho(predictions.reshape(observations.shape), observations)


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
    n_ahead = 1

    signal = lorenz96(K=K_signal, T=T + n_ahead, rng=rng)
    noise = np.empty((T + n_ahead, K_noise), dtype=np.float64)
    noise[0] = rng.standard_normal(K_noise)
    phi = rng.uniform(0.2, 0.8, size=K_noise)
    for t in range(1, T + n_ahead):
        noise[t] = phi * noise[t - 1] + rng.standard_normal(K_noise)

    full = np.concatenate([signal, noise], axis=1)
    order = rng.permutation(full.shape[1])
    # Predict the next-step value of L96 site 0 (original signal column 0).
    data = Dataset(X=full[:T, order], Y=signal[n_ahead : T + n_ahead, 0:1])
    informative_idx = {int(np.where(order == k)[0][0]) for k in range(K_signal)}
    print(f"Dataset: X{data.X.shape}, Y{data.Y.shape}")
    print(f"Informative columns (after shuffle): {sorted(informative_idx)}")

    fold1 = temporal_fold(data.X.shape[0], train_ratio=0.8)
    train = Subset(data, fold1.train)
    validation = Subset(data, fold1.validation)
    print(f"fold1 - train: {train.X.shape[0]}, validation: {validation.X.shape[0]}")

    E_range = list(range(1, 10 + 1))
    tau_range = list(range(1, 5 + 1))
    threshold = 0.1

    # Drop variables whose own embedding cannot predict the target manifold
    # even with the best (E, tau).
    def scan_one(i: int) -> float:
        return float(
            select(
                scan(train.X[:, i], train.Y, E=E_range, tau=tau_range, predict=simplex_projection, metric=_mean_rho),
                E=E_range,
                tau=tau_range,
            )[2]
        )

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        scores = np.fromiter(pool.map(scan_one, range(data.X.shape[1])), dtype=np.float64, count=data.X.shape[1])
        mask = scores >= threshold
        print(f"Retaining {int(mask.sum())} out of {data.X.shape[1]} variables after filtering with threshold={threshold}")

        kept = np.flatnonzero(mask).tolist()
        informative_idx = {new_idx for new_idx, old_idx in enumerate(kept) if old_idx in informative_idx}
        data = Dataset(X=data.X[:, mask], Y=data.Y)
        train = Subset(data, fold1.train)
        validation = Subset(data, fold1.validation)
        print(f"Informative columns after filter: {sorted(informative_idx)}")

        fold2 = temporal_fold(train.X.shape[0], train_ratio=0.75)
        print(f"fold2 - train: {fold2.train.shape[0]}, validation: {fold2.validation.shape[0]}")

        initial_context, plan = energy.holdout(
            data=train,
            fold=fold2,
            predict=simplex_projection,
            metric=mean_rho,
            batch_size=64,
        )

        def E(states: state.States, contexts: energy.Contexts) -> tuple[energy.Energies, energy.Contexts]:
            futures = [pool.submit(job) for job in plan(states, contexts)]
            n = states.shape[0]
            energies = np.empty(n, dtype=np.float64)
            new_contexts = np.empty((n, initial_context.shape[1]), dtype=np.float64)
            for f in futures:
                s, e, c = f.result()
                energies[s] = e
                new_contexts[s] = c
            return energies, new_contexts

        N = neighborhood.forward(data.X.shape[1])
        S = strategy.greedy(E, N)
        initial = strategy.Frontier(
            states=state.initial(),
            contexts=initial_context,
            energies=np.array([float("inf")], dtype=np.float64),
        )

        max_steps = K_signal + 2
        trace = list(strategy.run(initial, S, max_steps=max_steps, rng=rng))

    if not trace:
        print("No variables selected.")
        return

    predictions = np.zeros((len(trace), *validation.Y.shape))
    for j in range(len(trace)):
        selected = trace[j].states[0]
        predictions[j] = simplex_projection(train.X[:, selected], train.Y, validation.X[:, selected]).reshape(validation.Y.shape)
    validation_scores = mean_rho(predictions, np.broadcast_to(validation.Y, predictions.shape))

    selected = [int(i) for i in trace[-1].states[0]]
    print(f"\nSelected {len(selected)} variables (greedy):")
    for j, idx in enumerate(selected, 1):
        marker = "*" if idx in informative_idx else " "
        train_score = float(trace[j - 1].energies[0])
        val_score = float(validation_scores[j - 1])
        print(f"  {marker} dim {j}: idx={idx:3d}, train={train_score:.4f}, validation={val_score:.4f}")

    recovered = sum(1 for idx in selected if idx in informative_idx)
    print(f"\nRecovered {recovered}/{K_signal} informative variables in the first {len(selected)} selections.")
    print(f"Best held-out step: {int(np.argmin(validation_scores)) + 1} (score={float(validation_scores.min()):.4f})")


if __name__ == "__main__":
    main()
