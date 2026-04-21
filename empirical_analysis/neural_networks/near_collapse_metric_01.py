#!/usr/bin/env python3
"""
SFH-SGP_NEAR_COLLAPSE_METRIC_01
Test std(Dk) as continuous noise-level estimator
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

r = 3.5
n_total = 8000
n_burn = 2000

embedding_dim = 10
delay = 2
k_list = [2,4,8,16]

sigma_list = np.linspace(0, 0.01, 20)

def logistic(r, n):
    x = 0.5
    xs = []
    for i in range(n):
        x = r * x * (1 - x)
        xs.append(x)
    return np.array(xs)

def embed(x, dim, delay):
    N = len(x) - (dim - 1) * delay
    return np.array([x[i:i + dim*delay:delay] for i in range(N)])

def compute_dk(X, k):
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances = distances[:,1:] + 1e-10
    return np.mean(np.log(distances))

results = []

x = logistic(r, n_total)[n_burn:]

for sigma in sigma_list:
    
    x_noisy = x + sigma * np.random.randn(len(x))
    
    if np.isnan(x_noisy).any():
        raise ValueError("NaN detected")
    
    X = embed(x_noisy, embedding_dim, delay)
    
    Dk = []
    for k in k_list:
        Dk.append(compute_dk(X, k))
    
    std_Dk = np.std(Dk)
    
    results.append({
        "sigma": sigma,
        "std_Dk": std_Dk,
        "D2": Dk[0],
        "D4": Dk[1],
        "D8": Dk[2],
        "D16": Dk[3]
    })

df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("NEAR-COLLAPSE METRIC RESULTS")
print("=" * 60)
print(df.to_string(index=False))

# Analysis
print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)

sigma = df["sigma"].values
std_dk = df["std_Dk"].values

# Correlation
corr = np.corrcoef(sigma, std_dk)[0,1]
print(f"\nCorrelation (sigma vs std_Dk): {corr:.4f}")

# Check for monotonicity
diffs = np.diff(std_dk)
increasing = np.all(diffs >= 0) or np.sum(diffs > 0) > np.sum(diffs < 0)
print(f"Monotonically increasing: {increasing}")

# Check for smoothness (coefficient of variation)
cv = std_dk.std() / std_dk.mean()
print(f"Coefficient of variation: {cv:.4f}")

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

if corr > 0.9 and increasing:
    print("\n→ Case A: SMOOTH INCREASE")
    print("std(Dk) ∝ noise level")
    print("→ Created: continuous 'distance from periodicity' estimator")
elif corr > 0.5:
    print("\n→ Case B: MODERATE RELATIONSHIP")
    print("std(Dk) scales with noise but with variation")
    print("→ Useful: defines approximate threshold")
else:
    print("\n→ Case C: NO CLEAR RELATIONSHIP")
    print("std(Dk) does not track noise level")
    print("→ D(k) needs redesign")

df.to_csv("/home/student/sgp-tribe3/empirical_analysis/neural_networks/near_collapse_curve.csv", index=False)
print(f"\nSaved: near_collapse_curve.csv")