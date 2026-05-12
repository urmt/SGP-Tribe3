#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math

GLOBAL_SEED = 42
N = 128
DT = 0.05
STEPS = 4000
SIGMA = 1.0
K_VALUES = np.linspace(0.0, 6.0, 40)
WINDOW = 1000


def make_all_to_all(N):
    A = np.ones((N, N), dtype=np.float64)
    np.fill_diagonal(A, 0.0)
    deg = np.maximum(A.sum(axis=1), 1.0)
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


def run_transition_scan():
    A = make_all_to_all(N)
    results = []
    for K in K_VALUES:
        res = sim_kuramoto(N, K, A, seed=GLOBAL_SEED)
        R_final = float(res["R_final"])
        print(f"[K={K:6.3f}] R={R_final:.4f}")
        results.append((K, R_final))
    return results


def main():
    print(f"N={N}, steps={STEPS}, dt={DT}, sigma={SIGMA}")
    print(f"K range: [{K_VALUES[0]:.1f}, {K_VALUES[-1]:.1f}], {len(K_VALUES)} pts")
    print(f"Window: last {WINDOW} steps")
    print()

    rows = run_transition_scan()

    Ks = np.array([r[0] for r in rows])
    Rs = np.array([r[1] for r in rows])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Ks, Rs, "o-", ms=4, lw=1.2)
    ax.set_xlabel("K (coupling)")
    ax.set_ylabel("steady-state R")
    ax.set_title(f"Kuramoto Synchronization Transition (N={N})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("R_vs_K.png", dpi=150)
    plt.close()
    print()
    print("Saved R_vs_K.png")


if __name__ == "__main__":
    main()