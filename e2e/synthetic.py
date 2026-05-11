import numpy as np
from edmkit.metrics import mean_rho as _mean_rho
from edmkit.simplex_projection import simplex_projection
from edmkit.splits import Fold, temporal_fold

from edmkit.search import energy, neighborhood, state, strategy
from edmkit.search.dataset import Dataset, Subset


def mean_rho(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    return 1.0 - _mean_rho(predicted.reshape(observed.shape), observed)


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

    inner = temporal_fold(len(train), train_ratio=0.75)
    print(
        f"Inner fold - train: {len(inner.train)}, validation: {len(inner.validation)}"
    )

    E = energy.holdout(
        data=train,
        fold=Fold(train=inner.train, validation=inner.validation),
        predict=simplex_projection,
        metric=mean_rho,
    )
    N = neighborhood.forward(data.X.shape[1])
    step = strategy.greedy(E, N)
    initial = strategy.Frontier(
        states=state.initial(),
        contexts=E.initial(),
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
    print(
        f"\nRecovered {recovered}/{K_signal} informative variables "
        f"in the first {len(selected)} selections."
    )

    predictions = simplex_projection(
        train.X[:, selected], train.Y, validation.X[:, selected]
    )
    score = float(mean_rho(predictions, validation.Y))
    print(f"Held-out validation score (1 - mean_rho): {score:.4f}")


if __name__ == "__main__":
    main()
