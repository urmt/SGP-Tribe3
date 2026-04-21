#!/usr/bin/env python3
"""
SFH-SGP_OBSERVABLE_INVARIANCE_01
Test representation invariance of (α, DET)
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

def recurrence_matrix(x, epsilon):
    x = np.atleast_2d(x).T
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

def kuramoto_full(K, n, N=10):
    theta = np.random.rand(N) * 2*np.pi
    xs = []
    for _ in range(n):
        dtheta = np.zeros(N)
        for i in range(N):
            dtheta[i] = np.sum(np.sin(theta - theta[i]))
        theta += (K/N)*dtheta + 0.1*np.random.randn(N)
        xs.append(theta.copy())
    return np.array(xs)

np.random.seed(42)
results_repr = []

print("Running representation invariance test...")

for K in np.linspace(0.5, 2.5, 6):
    
    X = kuramoto_full(K, n_total)[n_burn:]
    X += sigma * np.random.randn(*X.shape)
    
    # 1. mean projection
    x_mean = np.mean(np.cos(X), axis=1)
    
    # 2. single oscillator
    x_single = X[:, 0]
    
    # 3. full state (flattened)
    x_full = X.reshape(len(X), -1)
    
    for label, x in [
        ("mean", x_mean),
        ("single", x_single),
        ("full", x_full)
    ]:
        alpha = compute_alpha(x, eps_list)
        R = recurrence_matrix(x, epsilon_det)
        det = compute_det(R)
        
        results_repr.append({
            "K": K,
            "repr": label,
            "alpha": alpha,
            "DET": det
        })

df = pd.DataFrame(results_repr)

print("\n" + "=" * 60)
print("REPRESENTATION INVARIANCE TEST")
print("=" * 60)
print(df.to_string(index=False))

print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)

for K in df["K"].unique():
    subset = df[df["K"] == K]
    print(f"\nK = {K:.1f}:")
    for _, row in subset.iterrows():
        print(f"  {row['repr']:<8}: α={row['alpha']:.3f}, DET={row['DET']:.3f}")

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

# Check invariance
mean_vals = df[df["repr"] == "mean"]["alpha"].values
single_vals = df[df["repr"] == "single"]["alpha"].values
full_vals = df[df["repr"] == "full"]["alpha"].values

diff_1 = np.abs(mean_vals - single_vals).mean()
diff_2 = np.abs(mean_vals - full_vals).mean()

print(f"\nMean vs Single: Δα = {diff_1:.3f}")
print(f"Mean vs Full: Δα = {diff_2:.3f}")

if diff_1 < 0.1 and diff_2 < 0.1:
    print("\n→ OBSERVABLE IS INVARIANT")
    print("→ Major result: α, DET describe intrinsic system state")
else:
    print("\n→ OBSERVABLE IS REPRESENTATION-DEPENDENT")
    print("→ Measures observed dynamics, not intrinsic state")

df.to_csv("/home/student/sgp-tribe3/empirical_analysis/neural_networks/representation_test.csv", index=False)
print(f"\nSaved: representation_test.csv")