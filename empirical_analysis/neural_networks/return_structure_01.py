#!/usr/bin/env python3
"""
SFH-SGP_RETURN_STRUCTURE_01
New observable: recurrence rate (not D(k) collapse)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

r_list = [3.5, 3.57, 3.9]
sigma_list = [0.0, 1e-4, 1e-3]

n_total = 6000
n_burn = 2000

epsilon_list = [1e-3, 5e-3, 1e-2]

def logistic(r, x0, n):
    x = x0
    xs = []
    for _ in range(n):
        x = r * x * (1 - x)
        xs.append(x)
    return np.array(xs)

def recurrence_rate(x, epsilon):
    x = x.reshape(-1, 1)
    D = pairwise_distances(x)
    N = len(x)
    mask = (D < epsilon) & (D > 0)
    return np.sum(mask) / (N * N - N)

results = []

print("Running return structure experiment...")

for r in r_list:
    for sigma in sigma_list:
        
        x0 = np.random.rand()
        x = logistic(r, x0, n_total)[n_burn:]
        x = x + sigma * np.random.randn(len(x))
        
        for eps in epsilon_list:
            rr = recurrence_rate(x, eps)
            
            results.append({
                "r": r,
                "sigma": sigma,
                "epsilon": eps,
                "recurrence_rate": rr
            })

df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("RECURRENCE RESULTS")
print("=" * 60)
print(df.to_string(index=False))

print("\n" + "=" * 60)
print("ANALYSIS BY REGIME")
print("=" * 60)

for r in r_list:
    subset = df[df["r"] == r]
    print(f"\nr = {r}:")
    for _, row in subset.iterrows():
        print(f"  σ={row['sigma']:.0e}, ε={row['epsilon']:.0e}: rr={row['recurrence_rate']:.6f}")

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

# Compare periodic vs chaotic
periodic_no_noise = df[(df["r"] == 3.5) & (df["sigma"] == 0.0)]
chaotic_no_noise = df[(df["r"] == 3.9) & (df["sigma"] == 0.0)]
periodic_noise = df[(df["r"] == 3.5) & (df["sigma"] > 0.0)]
chaotic_noise = df[(df["r"] == 3.9) & (df["sigma"] > 0.0)]

print(f"\nPeriodic (clean): {periodic_no_noise['recurrence_rate'].mean():.6f}")
print(f"Chaotic (clean): {chaotic_no_noise['recurrence_rate'].mean():.6f}")
print(f"Periodic (noisy): {periodic_noise['recurrence_rate'].mean():.6f}")
print(f"Chaotic (noisy): {chaotic_noise['recurrence_rate'].mean():.6f}")

if periodic_no_noise['recurrence_rate'].mean() > chaotic_no_noise['recurrence_rate'].mean():
    print("\n→ Recurrence separates periodic from chaotic (σ=0)")
else:
    print("\n→ No separation at σ=0")

if periodic_noise['recurrence_rate'].mean() > chaotic_noise['recurrence_rate'].mean():
    print("→ Recurrence separates periodic from chaotic (σ>0)")
else:
    print("→ No separation at σ>0")

df.to_csv("/home/student/sgp-tribe3/empirical_analysis/neural_networks/recurrence_results.csv", index=False)
print(f"\nSaved: recurrence_results.csv")