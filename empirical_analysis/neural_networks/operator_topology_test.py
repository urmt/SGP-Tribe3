#!/usr/bin/env python3
"""
SFH-SGP_OPERATOR_TOPOLOGY_01
Empirically probe whether operator space ℴ admits a local topology under Φ
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

# Kuramoto
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

def phi_distance(a1, d1, a2, d2):
    return np.sqrt((a1 - a2)**2 + (d1 - d2)**2)

# Parameter grid
lams = np.linspace(0, 1, 40)

# Compute Φ along path
alphas = []
dets = []

print("Computing Φ along parameter path...")

for lam in lams:
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
    
    alphas.append(alpha)
    dets.append(det)

alphas = np.array(alphas)
dets = np.array(dets)

# Compute local distances
param_dist = []
phi_dist = []

for i in range(1, len(lams)):
    d_param = abs(lams[i] - lams[i-1])
    d_phi = phi_distance(alphas[i], dets[i], alphas[i-1], dets[i-1])
    param_dist.append(d_param)
    phi_dist.append(d_phi)

param_dist = np.array(param_dist)
phi_dist = np.array(phi_dist)

# Correlation
corr = np.corrcoef(param_dist, phi_dist)[0,1]

print("\n" + "=" * 50)
print("TOPOLOGY TEST RESULTS")
print("=" * 50)
print(f"Mean Δparam: {np.mean(param_dist):.6f}")
print(f"Mean ΔΦ: {np.mean(phi_dist):.6f}")
print(f"Max ΔΦ: {np.max(phi_dist):.6f}")
print(f"Correlation (Δparam vs ΔΦ): {corr:.4f}")

# Interpretation
print("\n" + "=" * 50)
print("INTERPRETATION")
print("=" * 50)

if corr > 0.7 and np.max(phi_dist) < 0.1:
    interpretation = "PASS - Φ induces local metric structure"
elif corr > 0.3:
    interpretation = "WEAK - partial metric behavior"
else:
    interpretation = "FAIL - Φ is not metrically well-behaved"

print(f"\n{interpretation}")

# Summary file
summary = f"""TOPOLOGY RESULT 01
=====================

Correlation (Δparam vs ΔΦ): {corr:.4f}
Max ΔΦ: {np.max(phi_dist):.6f}
Mean ΔΦ: {np.mean(phi_dist):.6f}

INTERPRETATION: {interpretation}

EXPLANATION:
- If correlation > 0.7 and max ΔΦ < 0.1: Φ induces local metric structure on ℴ
- If correlation > 0.3 but max ΔΦ is larger: partial metric behavior  
- If correlation < 0.3: Φ is not metrically well-behaved under parameter variation

RESULT: {interpretation}
"""

with open("/home/student/sgp-tribe3/empirical_analysis/neural_networks/TOPOLOGY_RESULT_01.txt", "w") as f:
    f.write(summary)

print(f"\nSaved: TOPOLOGY_RESULT_01.txt")

df = pd.DataFrame({"lam": lams, "alpha": alphas, "DET": dets})
df.to_csv("/home/student/sgp-tribe3/empirical_analysis/neural_networks/operator_topology_path.csv", index=False)
print("Saved: operator_topology_path.csv")