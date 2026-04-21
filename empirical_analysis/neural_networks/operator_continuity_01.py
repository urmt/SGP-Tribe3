#!/usr/bin/env python3
"""
SFH-SGP_OPERATOR_CONTINUITY_01
Test continuity of Φ within parameterized operator families
"""
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

np.random.seed(42)

# Parameters
n_total = 3000
n_burn = 1000
sigma = 1e-3
eps_list = np.logspace(-4, -1, 10)
epsilon_det = 1e-2
l_min = 2

# Kuramoto system
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

K = 1.5
X = kuramoto_full(K, n_total)[n_burn:]
X += sigma * np.random.randn(*X.shape)

# Base observations
mean_obs = np.mean(np.cos(X), axis=1)
weighted_obs = np.dot(np.cos(X), np.linspace(1, 2, X.shape[1]))

# Recurrence functions
def recurrence_matrix(x, epsilon):
    x = np.atleast_2d(x).T if x.ndim == 1 else x
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

def delay_embed(x, m=3, tau=1):
    X = []
    for i in range(len(x) - (m-1)*tau):
        X.append([x[i + j*tau] for j in range(m)])
    return np.array(X)

def compute_features(obs):
    obs = np.atleast_2d(obs).T if obs.ndim == 1 else obs
    alpha = compute_alpha(obs, eps_list)
    R = recurrence_matrix(obs, epsilon_det)
    det = compute_det(R)
    return alpha, det

results = []

print("Running operator continuity test...")

# -----------------------------
# FAMILY 1: linear interpolation
# -----------------------------
print("\n1. Linear interpolation: mean ↔ weighted")
for lam in np.linspace(0, 1, 8):
    obs = lam * mean_obs + (1 - lam) * weighted_obs
    alpha, det = compute_features(obs)
    results.append({
        "family": "linear",
        "param": lam,
        "alpha": alpha,
        "DET": det
    })

# -----------------------------
# FAMILY 2: nonlinear strength
# -----------------------------
print("2. Nonlinear strength: sin(λx)")
for lam in np.linspace(0.5, 3.0, 8):
    obs = np.sin(lam * mean_obs)
    alpha, det = compute_features(obs)
    results.append({
        "family": "nonlinear",
        "param": lam,
        "alpha": alpha,
        "DET": det
    })

# -----------------------------
# FAMILY 3: embedding dimension
# -----------------------------
print("3. Embedding dimension: m = 2..10")
for m in range(2, 11):
    obs = delay_embed(mean_obs, m=m)
    alpha, det = compute_features(obs)
    results.append({
        "family": "embedding",
        "param": m,
        "alpha": alpha,
        "DET": det
    })

df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("OPERATOR CONTINUITY RESULTS")
print("=" * 60)

for family in df["family"].unique():
    subset = df[df["family"] == family].sort_values("param")
    print(f"\n{family.upper()}:")
    print(subset[["param", "alpha", "DET"]].to_string(index=False))

print("\n" + "=" * 60)
print("CONTINUITY ANALYSIS")
print("=" * 60)

for family in df["family"].unique():
    subset = df[df["family"] == family].sort_values("param")
    alpha_vals = subset["alpha"].values
    det_vals = subset["DET"].values
    
    # Compute gradients
    alpha_grad = np.diff(alpha_vals)
    det_grad = np.diff(det_vals)
    
    print(f"\n{family}:")
    print(f"  α range: {alpha_vals.min():.3f} → {alpha_vals.max():.3f}")
    print(f"  α gradient mean: {np.abs(alpha_grad).mean():.4f}")
    print(f"  α gradient max: {np.abs(alpha_grad).max():.4f}")
    print(f"  DET range: {det_vals.min():.3f} → {det_vals.max():.3f}")
    print(f"  DET gradient mean: {np.abs(det_grad).mean():.4f}")
    print(f"  DET gradient max: {np.abs(det_grad).max():.4f}")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)

# Check for smoothness
linear_df = df[df["family"] == "linear"]
max_linear_alpha_grad = np.abs(np.diff(linear_df["alpha"].values)).max()

nonlinear_df = df[df["family"] == "nonlinear"]
max_nonlinear_alpha_grad = np.abs(np.diff(nonlinear_df["alpha"].values)).max()

embed_df = df[df["family"] == "embedding"]
max_embed_alpha_grad = np.abs(np.diff(embed_df["alpha"].values)).max()

print(f"\nMax α jumps within families:")
print(f"  Linear interp: {max_linear_alpha_grad:.4f}")
print(f"  Nonlinear sin: {max_nonlinear_alpha_grad:.4f}")
print(f"  Embedding: {max_embed_alpha_grad:.4f}")

if max_linear_alpha_grad < 0.1 and max_nonlinear_alpha_grad < 0.1:
    print("\n→ CONTINUOUS within families")
elif max_embed_alpha_grad > 1.0:
    print("\n→ DISCONTINUOUS: embedding dimension causes major jumps")
else:
    print("\n→ PIECEWISE CONTINUOUS")

df.to_csv("/home/student/sgp-tribe3/empirical_analysis/neural_networks/operator_continuity.csv", index=False)
print(f"\nSaved: operator_continuity.csv")