#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math

GLOBAL_SEED = 42
DT = 0.05
STEPS = 2500
WINDOW = 500
TRIALS = 3
SIGMA = 1.0
N_VALUES = [16, 32, 64]
K_VALUES = np.linspace(0.0, 4.0, 20)


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
    return {"R_final": np.mean(R_trace[-WINDOW:])}


def main():
    norm = matplotlib.colors.Normalize(vmin=0, vmax=len(N_VALUES) - 1)
    cmap = plt.cm.viridis

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    for ni, N in enumerate(N_VALUES):
        A = make_all_to_all(N)
        results = []
        for K in K_VALUES:
            R_vals = []
            for trial in range(TRIALS):
                seed = GLOBAL_SEED + N * 1000 + int(K * 100) + trial * 77
                res = sim_kuramoto(N, K, A, seed=seed)
                R_vals.append(float(res["R_final"]))
            R_arr = np.array(R_vals)
            mean_R = float(np.mean(R_arr))
            chi = float(N * np.var(R_arr, ddof=1))
            print(f"[N={N:4d}] [K={K:5.3f}] R={mean_R:.4f} chi={chi:.4f}")
            results.append((K, mean_R, chi))

        ks = np.array([r[0] for r in results])
        Rs = np.array([r[1] for r in results])
        chis = np.array([r[2] for r in results])
        color = cmap(norm(ni))

        ax1.plot(ks, chis, "o-", color=color, ms=4, lw=1.2, label=f"N={N}")
        ax2.plot(ks, Rs, "o-", color=color, ms=4, lw=1.2, label=f"N={N}")

    ax1.set_xlabel("K (coupling)")
    ax1.set_ylabel("chi = N * Var(R)")
    ax1.set_title("Susceptibility vs K")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig("susceptibility_curves.png", dpi=150)
    plt.close(fig1)

    ax2.set_xlabel("K (coupling)")
    ax2.set_ylabel("mean R")
    ax2.set_title("Order Parameter vs K")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("order_parameter_curves.png", dpi=150)
    plt.close(fig2)

    print("Saved susceptibility_curves.png and order_parameter_curves.png")


if __name__ == "__main__":
    main()