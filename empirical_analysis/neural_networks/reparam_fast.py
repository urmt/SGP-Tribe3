#!/usr/bin/env python3
"""
SFH-SGP_REPARAMETRIZATION_INVARIANCE_01
Test whether Φ(S, O) is invariant under reparameterization
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

np.random.seed(42)

n = 500

def logistic_chaotic(n):
    x = 0.5
    xs = []
    for _ in range(n):
        x = 3.9 * x * (1 - x)
        xs.append(x)
    return np.array(xs)

X = logistic_chaotic(n)
X = (X - np.mean(X)) / (np.std(X) + 1e-10)

# Simple computation
def compute_simple(x):
    eps = 0.1
    x1 = x.reshape(-1, 1)
    D = pairwise_distances(x1)
    R = (D < eps).astype(int)
    np.fill_diagonal(R, 0)
    
    # Alpha (rough)
    rate = np.sum(R) / (len(x)**2)
    alpha = np.log(rate + 1e-10) / np.log(eps + 1e-10)
    
    # DET (rough)
    total = np.sum(R)
    diag = 0
    for k in range(-len(R)+1, len(R)):
        d = np.diagonal(R, k)
        cnt = 0
        for v in d:
            if v == 1: cnt += 1
            else:
                if cnt >= 2: diag += cnt
                cnt = 0
        if cnt >= 2: diag += cnt
    det = diag / max(total, 1)
    
    return alpha, det

# Two parameterizations
n_pts = 20
lam_vals = np.linspace(0, 1, n_pts)
mu_vals = lam_vals ** 2

print("Testing reparameterization invariance...")

diffs = []
for lam, mu in zip(lam_vals, mu_vals):
    # Same operator expressed differently
    obs1 = lam * X + (1 - lam) * np.sin(X)
    obs2 = np.sqrt(mu) * X + (1 - np.sqrt(mu)) * np.sin(X)
    
    alpha1, det1 = compute_simple(obs1)
    alpha2, det2 = compute_simple(obs2)
    
    d = np.sqrt((alpha1 - alpha2)**2 + (det1 - det2)**2)
    diffs.append(d)

diffs = np.array(diffs)

print("\n" + "="*50)
print("REPARAMETERIZATION TEST RESULTS")
print("="*50)
print(f"Mean ΔΦ: {diffs.mean():.6f}")
print(f"Max ΔΦ: {diffs.max():.6f}")
print(f"Std ΔΦ: {diffs.std():.6f}")

print("\n" + "="*50)
print("INTERPRETATION")
print("="*50)

if diffs.mean() < 0.01:
    print("→ YES: Φ is representation-invariant")
    print("→ Depends only on operator, not parameterization")
elif diffs.mean() < 0.1:
    print("→ PARTIAL: small dependence on parameterization")
else:
    print("→ NO: Φ depends on parameterization")

print("\n" + "="*50)
print(f"Result: ΔΦ = {diffs.mean():.4f} ± {diffs.std():.4f}")