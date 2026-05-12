#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
N_VALUES = [16, 32, 64]
K_VALUES = np.linspace(0.0, 4.0, 28)
TRIALS = 4
STEPS = 2000
DT = 0.05
SIGMA = 1.0


def make_all_to_all(N):
    A = np.ones((N, N), dtype=float)
    np.fill_diagonal(A, 0.0)
    return A


def sim_kuramoto(N, K, A, seed):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2*np.pi, N)
    omega = rng.normal(0, SIGMA, N)
    R_trace = np.zeros(STEPS)
    for t in range(STEPS):
        phase_diff = theta[None, :] - theta[:, None]
        coupling = np.sum(A * np.sin(phase_diff), axis=1)
        theta += DT * (omega + (K / N) * coupling)
        R = np.abs(np.mean(np.exp(1j * theta)))
        R_trace[t] = R
    return R_trace


def compute_metrics(N, R_trials):
    mean_R_per_trial = np.array([np.mean(r[-500:]) for r in R_trials])
    mean_R = np.mean(mean_R_per_trial)
    chi = N * np.var(mean_R_per_trial)
    m2 = np.mean(mean_R_per_trial ** 2)
    m4 = np.mean(mean_R_per_trial ** 4)
    binder = 1.0 - (m4 / (3.0 * (m2 ** 2 + 1e-12)))
    return mean_R, chi, binder


results = {}
for N in N_VALUES:
    print(f"\nRunning N={N}")
    A = make_all_to_all(N)
    R_curve, chi_curve, binder_curve = [], [], []
    for idx, K in enumerate(K_VALUES):
        print(f"  K={K:.2f} ({idx+1}/{len(K_VALUES)})")
        R_trials = []
        for trial in range(TRIALS):
            seed = SEED + N * 1000 + idx * 100 + trial
            R_trace = sim_kuramoto(N=N, K=K, A=A, seed=seed)
            R_trials.append(R_trace)
        mean_R, chi, binder = compute_metrics(N=N, R_trials=R_trials)
        R_curve.append(mean_R)
        chi_curve.append(chi)
        binder_curve.append(binder)
    results[N] = {"R": np.array(R_curve), "chi": np.array(chi_curve), "binder": np.array(binder_curve)}

plt.figure(figsize=(8,6))
for N in N_VALUES:
    plt.plot(K_VALUES, results[N]["R"], label=f"N={N}")
plt.xlabel("K")
plt.ylabel("Mean R")
plt.title("Finite-Size Sharpening")
plt.grid(True)
plt.legend()
plt.savefig("phase252_script5_order_parameter.png", dpi=200, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8,6))
for N in N_VALUES:
    plt.plot(K_VALUES, results[N]["chi"], label=f"N={N}")
plt.xlabel("K")
plt.ylabel("Susceptibility")
plt.title("Susceptibility Scaling")
plt.grid(True)
plt.legend()
plt.savefig("phase252_script5_susceptibility.png", dpi=200, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8,6))
for N in N_VALUES:
    plt.plot(K_VALUES, results[N]["binder"], label=f"N={N}")
plt.xlabel("K")
plt.ylabel("Binder")
plt.title("Binder Structure")
plt.grid(True)
plt.legend()
plt.savefig("phase252_script5_binder.png", dpi=200, bbox_inches="tight")
plt.close()

print("\n================================================")
print("SUMMARY")
print("================================================")
for N in N_VALUES:
    chi_curve = results[N]["chi"]
    peak_idx = np.argmax(chi_curve)
    print(f"N={N} | peak chi={chi_curve[peak_idx]:.4f} | K_peak={K_VALUES[peak_idx]:.3f}")
print("\nDONE")