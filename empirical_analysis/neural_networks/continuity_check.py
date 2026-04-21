#!/usr/bin/env python3
"""
SFH-SGP_CONTINUITY_RESOLUTION_01 - FAST VERSION
Only linear interpolation at high resolution
"""
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

np.random.seed(42)

n_total = 2000
n_burn = 500
sigma = 1e-3
eps_list = np.logspace(-4, -1, 6)
epsilon_det = 1e-2

def kuramoto_full(K, n, N=5):
    theta = np.random.rand(N) * 2*np.pi
    for _ in range(n):
        dtheta = np.zeros(N)
        for i in range(N):
            dtheta[i] = np.sum(np.sin(theta - theta[i]))
        theta += (K/N)*dtheta + 0.1*np.random.randn(N)
    return np.array([np.mean(np.cos(theta)) for _ in range(n)])

K = 1.5
X = kuramoto_full(K, n_total)[n_burn:]
X += sigma * (X - X.mean())

mean_obs = X
weighted_obs = np.convolve(X, [0.5, 0.5], mode='same')

def recurrence_matrix(x, eps):
    x = np.atleast_2d(x).T
    D = pairwise_distances(x)
    return (D < eps).astype(int)

def compute_alpha(x, eps_list):
    Rs = [np.sum(recurrence_matrix(x, e)) / (len(x)**2) for e in eps_list]
    log_R = np.log(np.array(Rs) + 1e-10)
    log_e = np.log(eps_list)
    return np.polyfit(log_e, log_R, 1)[0]

results = []

# Linear interpolation - 50 points
for lam in np.linspace(0, 1, 50):
    obs = lam * mean_obs + (1 - lam) * weighted_obs
    R = recurrence_matrix(obs, epsilon_det)
    diag = 0
    for k in range(-len(R)+1, len(R)):
        d = np.diagonal(R, k)
        cnt = 0
        for v in d:
            if v: cnt += 1
            else:
                if cnt >= 2: diag += cnt
                cnt = 0
        if cnt >= 2: diag += cnt
    det = diag / max(np.sum(R), 1)
    alpha = compute_alpha(obs, eps_list)
    
    results.append({"lam": lam, "alpha": alpha, "DET": det})

df = pd.DataFrame(results)

alpha_vals = df["alpha"].values
alpha_diff = np.abs(np.diff(alpha_vals))

global_range = alpha_vals.max() - alpha_vals.min()
max_jump = alpha_diff.max()
spikes = (alpha_diff > 0.1 * global_range).sum()

print("=" * 50)
print("LINEAR CONTINUITY (50 resolution)")
print("=" * 50)
print(f"α range: {alpha_vals.min():.4f} → {alpha_vals.max():.4f}")
print(f"Max jump: {max_jump:.4f}")
print(f"Mean jump: {alpha_diff.mean():.4f}")
print(f"Spikes (>10% range): {spikes}")

if max_jump < 0.05:
    print("✅ CONTINUOUS")
elif max_jump < 0.1:
    print("⚠️ PIECEWISE")
else:
    print("❌ DISCONTINUOUS")

df.to_csv("/home/student/sgp-tribe3/empirical_analysis/neural_networks/continuity_check.csv", index=False)