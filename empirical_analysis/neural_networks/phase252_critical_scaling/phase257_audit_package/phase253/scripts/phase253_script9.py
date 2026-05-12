#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
N_VALUES = [16, 32, 64]
K_VALUES = np.linspace(0.0, 4.0, 24)
TRIALS = 4
STEPS = 1800
DT = 0.05
P_ER = 0.25


def make_all_to_all(N):
    A = np.ones((N, N), dtype=float)
    np.fill_diagonal(A, 0.0)
    return A


def make_er_graph(N, p, rng):
    A = rng.random((N, N)) < p
    A = np.triu(A, 1)
    A = A + A.T
    return A.astype(float)


def make_omega(N, mode, rng):
    if mode == "gaussian":
        return rng.normal(0, 1.0, N)
    elif mode == "uniform":
        return rng.uniform(-1.0, 1.0, N)
    else:
        raise ValueError("bad omega mode")


def sim_kuramoto(N, K, A, omega, seed):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2*np.pi, N)
    R_trace = np.zeros(STEPS)
    deg = np.sum(A, axis=1)
    deg[deg == 0] = 1.0
    for t in range(STEPS):
        phase_diff = theta[None, :] - theta[:, None]
        coupling = np.sum(A * np.sin(phase_diff), axis=1)
        theta += DT * (omega + (K / deg) * coupling)
        R = np.abs(np.mean(np.exp(1j * theta)))
        R_trace[t] = R
    return R_trace


def compute_metrics(N, R_trials):
    mean_R_trials = np.array([np.mean(r[-400:]) for r in R_trials])
    mean_R = np.mean(mean_R_trials)
    chi = N * np.var(mean_R_trials)
    m2 = np.mean(mean_R_trials**2)
    m4 = np.mean(mean_R_trials**4)
    binder = 1.0 - (m4 / (3.0 * (m2**2 + 1e-12)))
    return mean_R, chi, binder


conditions = [
    ("all_to_all", "gaussian"),
    ("all_to_all", "uniform"),
    ("erdos_renyi", "gaussian"),
    ("erdos_renyi", "uniform"),
]

summary = []

for topo_mode, omega_mode in conditions:
    print(f"\n{'='*40}")
    print(topo_mode, omega_mode)
    print(f"{'='*40}")
    peak_chi_values = []
    binder_min_values = []
    for N in N_VALUES:
        rng = np.random.default_rng(SEED + N)
        if topo_mode == "all_to_all":
            A = make_all_to_all(N)
        else:
            A = make_er_graph(N, P_ER, rng)
        chi_curve = []
        binder_curve = []
        for k_idx, K in enumerate(K_VALUES):
            print(f"N={N} K={K:.2f}")
            R_trials = []
            for trial in range(TRIALS):
                local_rng = np.random.default_rng(SEED + N*1000 + k_idx*100 + trial)
                omega = make_omega(N, omega_mode, local_rng)
                R_trace = sim_kuramoto(N=N, K=K, A=A, omega=omega, seed=SEED + trial + k_idx)
                R_trials.append(R_trace)
            mean_R, chi, binder = compute_metrics(N, R_trials)
            chi_curve.append(chi)
            binder_curve.append(binder)
        chi_curve = np.array(chi_curve)
        binder_curve = np.array(binder_curve)
        peak_chi_values.append(float(np.max(chi_curve)))
        binder_min_values.append(float(np.min(binder_curve)))

    logN = np.log(N_VALUES)
    logChi = np.log(peak_chi_values)
    coeffs = np.polyfit(logN, logChi, 1)
    gamma = coeffs[0]
    fit = gamma * logN + coeffs[1]
    ss_res = np.sum((logChi - fit)**2)
    ss_tot = np.sum((logChi - np.mean(logChi))**2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))

    summary.append({
        "topology": topo_mode,
        "omega": omega_mode,
        "gamma": gamma,
        "r2": r2,
        "binder64": binder_min_values[-1]
    })

labels = [f"{s['topology']}\n{s['omega']}" for s in summary]
gammas = [s["gamma"] for s in summary]

plt.figure(figsize=(10, 6))
plt.bar(labels, gammas)
plt.ylabel("gamma")
plt.title("Universality Stress Test\nSusceptibility Scaling Exponent")
plt.grid(True)
plt.savefig("phase253_script9_universality.png", dpi=200, bbox_inches="tight")
plt.close()

print("\n" + "="*50)
print("UNIVERSALITY SUMMARY")
print("="*50)
for s in summary:
    print(f"{s['topology']} | {s['omega']} | gamma={s['gamma']:.4f} | R2={s['r2']:.4f} | binder64={s['binder64']:.4f}")
print("\nDONE")