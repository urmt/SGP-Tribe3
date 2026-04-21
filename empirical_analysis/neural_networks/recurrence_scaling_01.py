#!/usr/bin/env python3
"""
SFH-SGP_RECURRENCE_SCALING_01
Measure recurrence_rate scaling with epsilon
"""
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

r_list = [3.5, 3.57, 3.9]
sigma = 1e-3

epsilon_list = np.logspace(-4, -1, 15)

n_total = 6000
n_burn = 2000

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

print("Running recurrence scaling experiment...")

for r in r_list:
    
    x0 = np.random.rand()
    x = logistic(r, x0, n_total)[n_burn:]
    x = x + sigma * np.random.randn(len(x))
    
    for eps in epsilon_list:
        rr = recurrence_rate(x, eps)
        
        results.append({
            "r": r,
            "epsilon": eps,
            "recurrence_rate": rr
        })

df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("RECURRENCE SCALING RESULTS")
print("=" * 60)

pivot = df.pivot(index="epsilon", columns="r", values="recurrence_rate")
print(pivot.to_string())

print("\n" + "=" * 60)
print("SCALING ANALYSIS")
print("=" * 60)

for r in r_list:
    subset = df[df["r"] == r]
    eps = np.log(subset["epsilon"].values)
    rr = np.log(subset["recurrence_rate"].values)
    
    valid = np.isfinite(rr) & np.isfinite(eps)
    if valid.sum() > 2:
        alpha = np.polyfit(eps[valid], rr[valid], 1)[0]
        print(f"r={r}: α (scaling exponent) = {alpha:.3f}")
    else:
        print(f"r={r}: insufficient data for fit")

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

print("Looking for scaling law: recurrence_rate ~ ε^α")

print("\nKey observations:")
for r in r_list:
    subset = df[df["r"] == r]
    print(f"\nr={r}:")
    print(f"  ε range: {subset['epsilon'].min():.0e} to {subset['epsilon'].max():.0e}")
    print(f"  rr range: {subset['recurrence_rate'].min():.6f} to {subset['recurrence_rate'].max():.6f}")

df.to_csv("/home/student/sgp-tribe3/empirical_analysis/neural_networks/recurrence_scaling.csv", index=False)
print(f"\nSaved: recurrence_scaling.csv")