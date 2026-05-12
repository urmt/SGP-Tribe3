#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math

GLOBAL_SEED = 42
DT = 0.05
STEPS = 4000
WINDOW = 1000
SIGMA = 1.0
N_VALUES = [16, 32, 64, 128]
K_VALUES = np.linspace(0.0, 4.0, 30)


def make_all_to_all(N):
    A = np.ones((N, N), dtype=np.float64)
    np.fill_diagonal(A, 0.0)
    return A


def sim_kuramoto(N, K, A, steps=STEPS, dt=DT, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * math.pi, N)
    omega = rng.normal(0.0, SIGMA, N)
    R_trace = np.zeros(steps, dtype=np.float64)
    deg = np.maximum(A.sum(axis=1), 1.0)
    for t in range(steps):
        phase_diff = theta[None, :] - theta[:, None]
        coupling = K * np.sum(A * np.sin(phase_diff), axis=1) / deg
        theta = theta + dt * (omega + coupling)
        theta = np.mod(theta, 2.0 * math.pi)
        R_trace[t] = abs(np.mean(np.exp(1j * theta)))
    return {"R_trace": R_trace, "R_final": np.mean(R_trace[-WINDOW:])}


def main():
    fig, ax = plt.subplots(figsize=(8, 5))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=len(N_VALUES) - 1)
    cmap = plt.cm.viridis

    for ni, N in enumerate(N_VALUES):
        A = make_all_to_all(N)
        results = []
        for K in K_VALUES:
            res = sim_kuramoto(N, K, A, seed=GLOBAL_SEED + N)
            R_final = float(res["R_final"])
            print(f"[N={N:4d}] [K={K:5.3f}] R={R_final:.4f}")
            results.append(R_final)

        Rs = np.array(results)
        color = cmap(norm(ni))
        ax.plot(K_VALUES, Rs, "o-", color=color, ms=4, lw=1.2, label=f"N={N}")

    ax.set_xlabel("K (coupling)")
    ax.set_ylabel("steady-state R")
    ax.set_title("Finite-Size Sharpening of Kuramoto Transition")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("finite_size_transition.png", dpi=150)
    plt.close()
    print("Saved finite_size_transition.png")


if __name__ == "__main__":
    main()