#!/usr/bin/env python3
"""
SFH-SGP_CONTINUITY_RESOLUTION_01
High-resolution test for operator continuity
"""
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from itertools import repeat
from multiprocessing import Pool

np.random.seed(42)

# Parameters - NOT reduced
n_total = 4000
n_burn = 1000
sigma = 1e-3
eps_list = np.logspace(-4, -1, 8)
epsilon_det = 1e-2
l_min = 2

# Kuramoto system
def kuramoto_full(K, n, N=8):
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
    log_R = np.log(R_arr + 1e-10)
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

def compute_features(args):
    obs, = args
    obs = np.atleast_2d(obs).T if obs.ndim == 1 else obs
    alpha = compute_alpha(obs, eps_list)
    R = recurrence_matrix(obs, epsilon_det)
    det = compute_det(R)
    return alpha, det

def delay_embed(x, m=3, tau=1):
    X = []
    for i in range(len(x) - (m-1)*tau):
        X.append([x[i + j*tau] for j in range(m)])
    return np.array(X)

results = []

print("Running HIGH RESOLUTION continuity test...")

# -----------------------------
# FAMILY 1: Linear interpolation (50 points)
# -----------------------------
print("\n1. Linear interpolation: 50 λ values")
lam_vals = np.linspace(0, 1, 50)

for lam in lam_vals:
    obs = lam * mean_obs + (1 - lam) * weighted_obs
    alpha, det = compute_features((obs,))
    results.append({
        "family": "linear",
        "param": lam,
        "alpha": alpha,
        "DET": det
    })

# -----------------------------
# FAMILY 2: Nonlinear strength (50 points)
# ----------------------------
print("2. Nonlinear: 50 λ values")
lam_vals = np.linspace(0.3, 3.0, 50)

for lam in lam_vals:
    obs = np.sin(lam * mean_obs)
    alpha, det = compute_features((obs,))
    results.append({
        "family": "nonlinear",
        "param": lam,
        "alpha": alpha,
        "DET": det
    })

# -----------------------------
# FAMILY 3: Embedding dimension (15 points)
# ----------------------------
print("3. Embedding: m = 2..16")
for m in range(2, 17):
    obs = delay_embed(mean_obs, m=m)
    alpha, det = compute_features((obs,))
    results.append({
        "family": "embedding",
        "param": m,
        "alpha": alpha,
        "DET": det
    })

df = pd.DataFrame(results)

# -----------------------------
# NUMERICAL CONTINUITY CHECK
# -----------------------------
print("\n" + "=" * 60)
print("HIGH-RESOLUTION CONTINUITY ANALYSIS")
print("=" * 60)

for family in df["family"].unique():
    subset = df[df["family"] == family].sort_values("param")
    alpha_vals = subset["alpha"].values
    det_vals = subset["DET"].values
    
    # Compute jumps (differences)
    alpha_diff = np.abs(np.diff(alpha_vals, prepend=alpha_vals[0]))
    det_diff = np.abs(np.diff(det_vals, prepend=det_vals[0]))
    
    print(f"\n{family}:")
    print(f"  α range: {alpha_vals.min():.4f} → {alpha_vals.max():.4f}")
    print(f"  α max jump: {alpha_diff.max():.4f}")
    print(f"  α mean jump: {alpha_diff.mean():.4f}")
    print(f"  α jumps > 0.1: {(alpha_diff > 0.1).sum()}")
    print(f"  DET range: {det_vals.min():.4f} → {det_vals.max():.4f}")
    print(f"  DET max jump: {det_diff.max():.4f}")
    print(f"  DET mean jump: {det_diff.mean():.4f}")

print("\n" + "=" * 60)
print("GLOBAL CONTINUITY ASSESSMENT")
print("=" * 60)

for family in df["family"].unique():
    subset = df[df["family"] == family].sort_values("param")
    alpha_vals = subset["alpha"].values
    alpha_diff = np.abs(np.diff(alpha_vals))
    
    global_range = alpha_vals.max() - alpha_vals.min()
    max_jump = alpha_diff.max()
    spikes = (alpha_diff > 0.1 * global_range).sum()
    
    if max_jump < 0.05 and spikes == 0:
        status = "✅ CONTINUOUS"
    elif max_jump < 0.1 and spikes < 2:
        status = "⚠️ PIECEWISE CONTINUOUS"
    else:
        status = "❌ DISCONTINUOUS"
    
    print(f"\n{family}: {status}")
    print(f"  Global α range: {global_range:.4f}")
    print(f"  Max local jump: {max_jump:.4f}")
    print(f"  Spike count (jump > 10% range): {spikes}")

df.to_csv("/home/student/sgp-tribe3/empirical_analysis/neural_networks/continuity_high_res.csv", index=False)
print(f"\nSaved: continuity_high_res.csv")