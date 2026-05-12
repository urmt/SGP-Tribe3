#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

N_VALUES = [16, 32, 64]
K_VALUES = np.linspace(0.4, 3.0, 18)
TRIALS = 4
STEPS = 1000
TRANSIENT = 300
DT = 0.05

rng_global = np.random.default_rng(42)
N_MAX = N_VALUES[-1]
omega = rng_global.normal(0, 1.0, N_MAX)
A_full = np.ones((N_MAX, N_MAX), dtype=float)
np.fill_diagonal(A_full, 0.0)
deg_full = np.sum(A_full, axis=1)


def sim(N, K, seed):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2*np.pi, N)
    R_trace = np.zeros(STEPS)
    for t in range(STEPS):
        phase_diff = theta[None, :] - theta[:, None]
        coupling = np.sum(A_full[:N, :N] * np.sin(phase_diff), axis=1)
        theta += DT * (omega[:N] + (K / deg_full[:N]) * coupling)
        R_trace[t] = np.abs(np.mean(np.exp(1j * theta)))
    return R_trace


data = {}
for N in N_VALUES:
    print(f"N={N}")
    data[N] = {"R": np.zeros((len(K_VALUES), TRIALS)), "tau": np.zeros((len(K_VALUES), TRIALS))}
    for ki, K in enumerate(K_VALUES):
        for trial in range(TRIALS):
            seed = 42 + N * 1000 + ki * 100 + trial * 77
            R_trace = sim(N, K, seed)
            data[N]["R"][ki, trial] = float(np.mean(R_trace[TRANSIENT:]))
            R_final = float(np.mean(R_trace[-100:]))
            found = False
            for t in range(STEPS - 50):
                if np.all(np.abs(R_trace[t:t + 50] - R_final) < 0.02):
                    data[N]["tau"][ki, trial] = t * DT
                    found = True
                    break
            if not found:
                data[N]["tau"][ki, trial] = STEPS * DT

for N in N_VALUES:
    R_mean = np.mean(data[N]["R"], axis=1)
    chi = N * np.var(data[N]["R"], axis=1)
    R2 = np.mean(data[N]["R"] ** 2, axis=1)
    R4 = np.mean(data[N]["R"] ** 4, axis=1)
    binder = 1.0 - R4 / (3.0 * (R2 ** 2 + 1e-12))
    tau_mean = np.mean(data[N]["tau"], axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(K_VALUES, R_mean, "o-")
    plt.xlabel("K")
    plt.ylabel("mean R")
    plt.title(f"Order Parameter N={N}")
    plt.grid(True)
    plt.savefig(f"phase256_replication_order_N{N}.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(K_VALUES, chi, "o-")
    plt.xlabel("K")
    plt.ylabel("chi")
    plt.title(f"Susceptibility N={N}")
    plt.grid(True)
    plt.savefig(f"phase256_replication_chi_N{N}.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(K_VALUES, binder, "o-")
    plt.xlabel("K")
    plt.ylabel("Binder")
    plt.title(f"Binder Cumulant N={N}")
    plt.grid(True)
    plt.savefig(f"phase256_replication_binder_N{N}.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(K_VALUES, tau_mean, "o-")
    plt.xlabel("K")
    plt.ylabel("tau")
    plt.title(f"Relaxation Time N={N}")
    plt.grid(True)
    plt.savefig(f"phase256_replication_tau_N{N}.png", dpi=150, bbox_inches="tight")
    plt.close()

    chi_peak = float(np.max(chi))
    k_chi_peak = float(K_VALUES[np.argmax(chi)])
    binder_min = float(np.min(binder))
    k_binder_min = float(K_VALUES[np.argmin(binder)])
    tau_peak = float(np.max(tau_mean))
    k_tau_peak = float(K_VALUES[np.argmax(tau_mean)])

    with open("phase256_replication_summary.txt", "a") as f:
        f.write(f"N={N}\n")
        f.write(f"  chi_peak={chi_peak:.4f} K_chi_peak={k_chi_peak:.4f}\n")
        f.write(f"  binder_min={binder_min:.4f} K_binder_min={k_binder_min:.4f}\n")
        f.write(f"  tau_peak={tau_peak:.4f} K_tau_peak={k_tau_peak:.4f}\n")

print("DONE")