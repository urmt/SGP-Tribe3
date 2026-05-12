#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
N = 64
K_FORWARD = np.linspace(0.0, 4.0, 32)
K_BACKWARD = np.linspace(4.0, 0.0, 32)
STEPS = 1600
DT = 0.05
TRIALS = 4

A = np.ones((N, N), dtype=float)
np.fill_diagonal(A, 0.0)
deg = np.sum(A, axis=1)

rng_global = np.random.default_rng(SEED)
omega = rng_global.normal(0, 1.0, N)


def run_sweep(K_values, theta_init):
    theta = theta_init.copy()
    R_curve = []
    for K in K_values:
        R_trace = []
        for _ in range(STEPS):
            phase_diff = theta[None, :] - theta[:, None]
            coupling = np.sum(A * np.sin(phase_diff), axis=1)
            theta += DT * (omega + (K / deg) * coupling)
            R = np.abs(np.mean(np.exp(1j * theta)))
            R_trace.append(R)
        R_mean = np.mean(R_trace[-300:])
        R_curve.append(R_mean)
    return np.array(R_curve), theta


forward_trials = []
backward_trials = []

for trial in range(TRIALS):
    rng = np.random.default_rng(SEED + trial)
    theta0 = rng.uniform(0, 2*np.pi, N)
    R_forward, theta_final = run_sweep(K_FORWARD, theta0)
    R_backward, _ = run_sweep(K_BACKWARD, theta_final)
    forward_trials.append(R_forward)
    backward_trials.append(R_backward)

forward_trials = np.array(forward_trials)
backward_trials = np.array(backward_trials)
mean_forward = np.mean(forward_trials, axis=0)
mean_backward = np.mean(backward_trials, axis=0)
mean_backward_flip = mean_backward[::-1]

delta = np.abs(mean_forward - mean_backward_flip)
loop_area = np.trapezoid(delta, K_FORWARD)
max_gap = np.max(delta)
mean_gap = np.mean(delta)

plt.figure(figsize=(8,6))
plt.plot(K_FORWARD, mean_forward, label="forward sweep")
plt.plot(K_FORWARD, mean_backward_flip, label="backward sweep")
plt.xlabel("K")
plt.ylabel("R")
plt.title("Phase 254\nHysteresis Test")
plt.legend()
plt.grid(True)
plt.savefig("phase254_script10_hysteresis.png", dpi=200, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8,6))
plt.plot(K_FORWARD, delta)
plt.xlabel("K")
plt.ylabel("|R_forward - R_backward|")
plt.title("Phase 254\nSweep Difference")
plt.grid(True)
plt.savefig("phase254_script10_gap.png", dpi=200, bbox_inches="tight")
plt.close()

print("\n========================================")
print("PHASE 254 — HYSTERESIS TEST")
print("========================================")
print(f"loop_area = {loop_area:.6f}")
print(f"max_gap   = {max_gap:.6f}")
print(f"mean_gap  = {mean_gap:.6f}")
verdict = "CONTINUOUS_TRANSITION" if loop_area < 0.15 else "HYSTERETIC_OR_FIRST_ORDER"
print(f"\nVERDICT: {verdict}")
print("\nFILES:")
print("phase254_script10_hysteresis.png")
print("phase254_script10_gap.png")