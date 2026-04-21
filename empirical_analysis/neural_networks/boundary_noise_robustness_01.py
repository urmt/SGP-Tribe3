#!/usr/bin/env python3
"""
SFH-SGP_BOUNDARY_NOISE_ROBUSTNESS_01
Map collapse probability as function of (r, sigma) near chaos boundary
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

r_list = [3.55, 3.56, 3.565, 3.568, 3.570, 3.575, 3.58]
sigma_list = [0.0, 1e-4, 5e-4, 1e-3, 5e-3]

n_trials = 20
n_total = 6000
n_burn = 2000

embedding_dim = 10
delay = 2
k_list = [2,4,8,16]

def logistic(r, x0, n):
    x = x0
    xs = []
    for _ in range(n):
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

print("Running boundary-noise sweep...")

for r in r_list:
    for sigma in sigma_list:
        
        collapse_flags = []
        
        for trial in range(n_trials):
            
            x0 = np.random.rand()
            x = logistic(r, x0, n_total)[n_burn:]
            
            x = x + sigma * np.random.randn(len(x))
            
            if np.isnan(x).any():
                raise ValueError("NaN detected")
            
            X = embed(x, embedding_dim, delay)
            
            Dk = [compute_dk(X, k) for k in k_list]
            std_Dk = np.std(Dk)
            
            collapse = std_Dk < 0.05
            collapse_flags.append(collapse)
        
        collapse_rate = np.mean(collapse_flags)
        
        results.append({
            "r": r,
            "sigma": sigma,
            "collapse_rate": collapse_rate
        })

df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("BOUNDARY NOISE RESULTS")
print("=" * 60)

pivot = df.pivot(index="r", columns="sigma", values="collapse_rate")
print(pivot.to_string())

print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)

r_vals = df["r"].unique()
sigma_vals = df["sigma"].unique()

for sigma in sigma_vals:
    subset = df[df["sigma"] == sigma]
    rates = subset["collapse_rate"].values
    print(f"σ={sigma:.0e}: rates = {rates}")

df.to_csv("/home/student/sgp-tribe3/empirical_analysis/neural_networks/boundary_noise_results.csv", index=False)

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

baseline = df[df["sigma"] == 0.0]["collapse_rate"].values
noisy_rates = df[df["sigma"] > 0.0].groupby("r")["collapse_rate"].mean().values

print(f"\nBaseline (σ=0) collapse rates: {baseline}")
print(f"Noisy mean collapse rates: {noisy_rates}")

if np.allclose(baseline, noisy_rates, atol=0.1):
    print("\n→ Boundary is STABLE under noise")
    print("→ D(k) detects intrinsic dynamical structure")
elif np.any(noisy_rates < baseline):
    print("\n→ Boundary SHIFTS with noise")
    print("→ Collapse probability degrades under perturbation")
else:
    print("\n→ Mixed or unclear behavior")

print(f"\nSaved: boundary_noise_results.csv")