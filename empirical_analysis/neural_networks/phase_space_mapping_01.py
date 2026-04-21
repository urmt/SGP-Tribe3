#!/usr/bin/env python3
"""
SFH-SGP_PHASE_SPACE_MAPPING_01
Map continuous parameter sweeps into (α, DET) space
"""
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

n_total = 3000
n_burn = 1000
sigma = 1e-3
eps_list = np.logspace(-4, -1, 8)
epsilon_det = 1e-2
l_min = 2

def logistic(r, x0, n):
    x = x0
    xs = []
    for _ in range(n):
        x = r * x * (1 - x)
        xs.append(x)
    return np.array(xs)

def ou_process(theta, n):
    x = 0
    xs = []
    for _ in range(n):
        x = x + theta * (-x) + np.random.randn()*0.1
        xs.append(x)
    return np.array(xs)

def kuramoto(K, n, N=10):
    theta = np.random.rand(N) * 2*np.pi
    xs = []
    for _ in range(n):
        dtheta = np.zeros(N)
        for i in range(N):
            dtheta[i] = np.sum(np.sin(theta - theta[i]))
        theta += (K/N)*dtheta + 0.1*np.random.randn(N)
        xs.append(np.mean(np.cos(theta)))
    return np.array(xs)

def recurrence_matrix(x, epsilon):
    x = x.reshape(-1,1)
    D = pairwise_distances(x)
    R = (D < epsilon).astype(int)
    np.fill_diagonal(R, 0)
    return R

def recurrence_rate(R):
    N = R.shape[0]
    return np.sum(R) / (N*N - N)

def compute_alpha(x, eps_list):
    R_vals = []
    for eps in eps_list:
        R = recurrence_matrix(x, eps)
        R_vals.append(recurrence_rate(R))
    R_arr = np.array(R_vals)
    log_R = np.log(R_arr)
    log_eps = np.log(eps_list)
    valid = np.isfinite(log_R)
    if valid.sum() > 2:
        return np.polyfit(log_eps[valid], log_R[valid], 1)[0]
    return np.nan

def compute_det(R, l_min=2):
    N = R.shape[0]
    diag_counts = 0
    total_rec = np.sum(R)
    for k in range(-N+1, N):
        diag = np.diagonal(R, offset=k)
        length = 0
        for val in diag:
            if val == 1:
                length += 1
            else:
                if length >= l_min:
                    diag_counts += length
                length = 0
        if length >= l_min:
            diag_counts += length
    return diag_counts / total_rec if total_rec > 0 else 0

np.random.seed(42)
results = []

print("Running phase space mapping...")

# Logistic sweep
print("  Logistic: r = 3.4 → 4.0")
for r in np.linspace(3.4, 4.0, 15):
    x = logistic(r, np.random.rand(), n_total)[n_burn:]
    x += sigma * np.random.randn(len(x))
    alpha = compute_alpha(x, eps_list)
    R = recurrence_matrix(x, epsilon_det)
    det = compute_det(R)
    results.append({"system": "logistic", "param": r, "alpha": alpha, "DET": det})

# Kuramoto sweep
print("  Kuramoto: K = 0 → 3")
for K in np.linspace(0.0, 3.0, 10):
    x = kuramoto(K, n_total)[n_burn:]
    x += sigma * np.random.randn(len(x))
    alpha = compute_alpha(x, eps_list)
    R = recurrence_matrix(x, epsilon_det)
    det = compute_det(R)
    results.append({"system": "kuramoto", "param": K, "alpha": alpha, "DET": det})

# OU sweep
print("  OU: θ = 0.1 → 2.0")
for theta in np.linspace(0.1, 2.0, 10):
    x = ou_process(theta, n_total)[n_burn:]
    x += sigma * np.random.randn(len(x))
    alpha = compute_alpha(x, eps_list)
    R = recurrence_matrix(x, epsilon_det)
    det = compute_det(R)
    results.append({"system": "ou", "param": theta, "alpha": alpha, "DET": det})

df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("PHASE SPACE MAPPING RESULTS")
print("=" * 60)

for sys in ["logistic", "kuramoto", "ou"]:
    subset = df[df["system"] == sys]
    print(f"\n{sys.upper()}:")
    print(f"  α range: {subset['alpha'].min():.3f} → {subset['alpha'].max():.3f}")
    print(f"  DET range: {subset['DET'].min():.3f} → {subset['DET'].max():.3f}")
    print(f"  Parameters: {subset['param'].min():.2f} → {subset['param'].max():.2f}")

print("\n" + "=" * 60)
print("GEOMETRIC STRUCTURE")
print("=" * 60)

logistic_df = df[df["system"] == "logistic"]
kuramoto_df = df[df["system"] == "kuramoto"]
ou_df = df[df["system"] == "ou"]

print("\nLogistic (r near 3.57, chaos onset):")
boundary = logistic_df[(logistic_df["param"] > 3.55) & (logistic_df["param"] < 3.60)]
print(boundary[["param", "alpha", "DET"]].to_string(index=False))

print("\nKuramoto (K near 1.0, sync boundary):")
k_boundary = kuramoto_df[(kuramoto_df["param"] > 0.8) & (kuramoto_df["param"] < 1.2)]
print(k_boundary[["param", "alpha", "DET"]].to_string(index=False))

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

print("\nExpected:")
print("  Logistic → sharp transition (discrete)")
print("  Kuramoto → smooth curve (continuous)")
print("  OU → smooth diffusion (stochastic)")

df.to_csv("/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase_space_mapping.csv", index=False)
print(f"\nSaved: phase_space_mapping.csv")