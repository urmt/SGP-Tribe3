#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
N = 64
K_VALUES = np.linspace(0.2, 3.5, 28)
DT = 0.05
STEPS = 3000
TRIALS = 4

rng_global = np.random.default_rng(SEED)
omega = rng_global.normal(0, 1.0, N)

A = np.ones((N, N), dtype=float)
np.fill_diagonal(A, 0.0)
deg = np.sum(A, axis=1)


def simulate(K, theta0):
    theta = theta0.copy()
    R_trace = []
    for _ in range(STEPS):
        phase_diff = theta[None, :] - theta[:, None]
        coupling = np.sum(A * np.sin(phase_diff), axis=1)
        theta += DT * (omega + (K / deg) * coupling)
        R = np.abs(np.mean(np.exp(1j * theta)))
        R_trace.append(R)
    return np.array(R_trace)


tau_mean = []
tau_std = []

for K in K_VALUES:
    tau_trials = []
    for trial in range(TRIALS):
        rng = np.random.default_rng(SEED + trial)
        theta0 = rng.uniform(0, 2*np.pi, N)
        R_trace = simulate(K, theta0)
        R_eq = np.mean(R_trace[-300:])
        threshold = 0.9 * R_eq
        idx = np.where(R_trace >= threshold)[0]
        if len(idx) == 0:
            tau = STEPS * DT
        else:
            tau = idx[0] * DT
        tau_trials.append(tau)
    tau_mean.append(np.mean(tau_trials))
    tau_std.append(np.std(tau_trials))

tau_mean = np.array(tau_mean)
tau_std = np.array(tau_std)

peak_idx = np.argmax(tau_mean)
K_peak = K_VALUES[peak_idx]
tau_peak = tau_mean[peak_idx]

plt.figure(figsize=(8,6))
plt.errorbar(K_VALUES, tau_mean, yerr=tau_std, capsize=3)
plt.axvline(K_peak, linestyle="--")
plt.xlabel("K")
plt.ylabel("Relaxation time tau")
plt.title("Phase 255\nCritical Slowing Down")
plt.grid(True)
plt.savefig("phase255_script11_critical_slowing.png", dpi=200, bbox_inches="tight")
plt.close()

print("\n========================================")
print("PHASE 255 — CRITICAL SLOWING")
print("========================================")
print(f"K_peak   = {K_peak:.4f}")
print(f"tau_peak = {tau_peak:.4f}")
baseline = np.mean(tau_mean[:5])
ratio = tau_peak / baseline
print(f"baseline_tau = {baseline:.4f}")
print(f"peak_ratio   = {ratio:.4f}")
verdict = "CRITICAL_SLOWING_CONFIRMED" if ratio > 2.0 else "NO_CLEAR_SLOWING"
print(f"\nVERDICT: {verdict}")
print("\nFILES:")
print("phase255_script11_critical_slowing.png")