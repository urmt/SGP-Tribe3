#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

N_VALUES = [16, 32, 64]
K_VALUES = np.linspace(0.4, 3.0, 20)
STEPS = 1200
TRANSIENT = 400
TRIALS_K = 3
TRIALS_TAU = 1
DT = 0.02

rng_global = np.random.default_rng(42)
N_MAX = N_VALUES[-1]
omega_full = rng_global.normal(0, 1.0, N_MAX)
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
        theta += DT * (omega_full[:N] + (K / deg_full[:N]) * coupling)
        R_trace[t] = np.abs(np.mean(np.exp(1j * theta)))
    return R_trace


results = {}

for N in N_VALUES:
    print(f"N={N}")
    chi_arr = np.zeros(len(K_VALUES))
    binder_arr = np.zeros(len(K_VALUES))
    tau_arr = np.zeros(len(K_VALUES))
    for ki, K in enumerate(K_VALUES):
        R_late_k = []
        tau_k = []
        for trial in range(TRIALS_K):
            seed = 42 + N * 1000 + ki * 100 + trial * 77
            R_trace = sim(N, K, seed)
            R_late = np.mean(R_trace[TRANSIENT:])
            R_late_k.append(float(R_late))
        R_arr = np.array(R_late_k, dtype=float)
        chi_arr[ki] = float(N * np.var(R_arr))
        m2 = float(np.mean(R_arr ** 2))
        m4 = float(np.mean(R_arr ** 4))
        binder_arr[ki] = float(1.0 - m4 / (3.0 * (m2 ** 2 + 1e-12)))
        for trial in range(TRIALS_TAU):
            seed = 42 + N * 2000 + ki * 100 + trial * 33
            R_trace = sim(N, K, seed)
            R_final = float(np.mean(R_trace[-200:]))
            for step in range(STEPS):
                if abs(R_trace[step] - R_final) < 0.05:
                    tau_k.append(step * DT)
                    break
            else:
                tau_k.append(STEPS * DT)
        tau_arr[ki] = float(np.mean(tau_k))

    k_chi_peak = float(K_VALUES[np.argmax(chi_arr)])
    k_binder_min = float(K_VALUES[np.argmin(binder_arr)])
    k_tau_peak = float(K_VALUES[np.argmax(tau_arr)])

    results[N] = {
        "k_chi_peak": k_chi_peak,
        "k_binder_min": k_binder_min,
        "k_tau_peak": k_tau_peak,
    }
    print(f"  K_chi={k_chi_peak:.3f} K_binder={k_binder_min:.3f} K_tau={k_tau_peak:.3f}")

invN = 1.0 / np.array(N_VALUES, dtype=float)

for label, key in [("kchi", "k_chi_peak"), ("kbinder", "k_binder_min"), ("ktau", "k_tau_peak")]:
    y_vals = np.array([results[N][key] for N in N_VALUES], dtype=float)
    coeffs = np.polyfit(invN, y_vals, 1)
    ss_res = np.sum((y_vals - (coeffs[0] * invN + coeffs[1])) ** 2)
    ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    print(f"{key}: Kc_inf={coeffs[0]:.4f} R2={r2:.4f}")
    plt.figure(figsize=(8, 6))
    plt.plot(invN, y_vals, "o-")
    plt.plot(invN, coeffs[0] * invN + coeffs[1], "--")
    plt.xlabel("1/N")
    plt.ylabel(label)
    plt.title(f"{label} vs 1/N")
    plt.grid(True)
    plt.savefig(f"phase256_script12_{label}.png", dpi=200, bbox_inches="tight")
    plt.close()

plt.figure(figsize=(8, 6))
for key, color, marker in [("k_chi_peak", "blue", "o"), ("k_binder_min", "green", "s"), ("k_tau_peak", "red", "^")]:
    y_vals = np.array([results[N][key] for N in N_VALUES], dtype=float)
    plt.plot(invN, y_vals, marker=marker, color=color, label=key)
    coeffs = np.polyfit(invN, y_vals, 1)
    plt.plot(invN, coeffs[0] * invN + coeffs[1], "--", color=color)
plt.xlabel("1/N")
plt.ylabel("K_estimate")
plt.title("Critical Point Extrapolation")
plt.legend()
plt.grid(True)
plt.savefig("phase256_script12_combined.png", dpi=200, bbox_inches="tight")
plt.close()

Kc_chi = float(np.polyfit(invN, np.array([results[N]["k_chi_peak"] for N in N_VALUES]), 1)[0])
Kc_binder = float(np.polyfit(invN, np.array([results[N]["k_binder_min"] for N in N_VALUES]), 1)[0])
Kc_tau = float(np.polyfit(invN, np.array([results[N]["k_tau_peak"] for N in N_VALUES]), 1)[0])

max_deviation = float(max(abs(Kc_chi - Kc_binder), abs(Kc_chi - Kc_tau), abs(Kc_binder - Kc_tau)))
verdict = "UNIFIED_CRITICAL_POINT_CONFIRMED" if max_deviation < 0.25 else "CRITICAL_ESTIMATORS_INCONSISTENT"

print("\nRESULTS:")
for N in N_VALUES:
    print(f"N={N} K_chi={results[N]['k_chi_peak']:.4f} K_binder={results[N]['k_binder_min']:.4f} K_tau={results[N]['k_tau_peak']:.4f}")
print("\nEXTRAPOLATED_KC:")
print(f"Kc_chi={Kc_chi:.4f}")
print(f"Kc_binder={Kc_binder:.4f}")
print(f"Kc_tau={Kc_tau:.4f}")
print("\nMAX_DEVIATION:")
print(f"{max_deviation:.4f}")
print("\nVERDICT:")
print(verdict)