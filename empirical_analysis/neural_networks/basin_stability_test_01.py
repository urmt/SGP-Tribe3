#!/usr/bin/env python3
"""
SFH-SGP_BASIN_STABILITY_TEST_01
Test D(k) collapse across basin of attraction
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

r = 3.5
n_total = 6000
n_burn = 2000

embedding_dim = 10
delay = 2
k_list = [2,4,8,16]

n_trials = 50

def logistic(r, x0, n):
    x = x0
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

print("Testing basin stability...")

for trial in range(n_trials):
    
    x0 = np.random.rand()
    
    x = logistic(r, x0, n_total)[n_burn:]
    
    if np.isnan(x).any():
        raise ValueError("NaN detected")
    
    X = embed(x, embedding_dim, delay)
    
    Dk = []
    for k in k_list:
        Dk.append(compute_dk(X, k))
    
    std_Dk = np.std(Dk)
    collapse = std_Dk < 0.05
    
    results.append({
        "trial": trial,
        "x0": x0,
        "std_Dk": std_Dk,
        "collapse": collapse
    })

df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("BASIN STABILITY RESULTS")
print("=" * 60)
print(df.to_string(index=False))

collapse_rate = df["collapse"].mean()
std_Dk_mean = df["std_Dk"].mean()
std_Dk_std = df["std_Dk"].std()

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Collapse rate: {collapse_rate:.3f}")
print(f"std(Dk) mean: {std_Dk_mean:.6f}")
print(f"std(Dk) std: {std_Dk_std:.6f}")

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

if collapse_rate >= 0.95:
    print("\n→ Case A: COLLAPSE IS BASIN-LEVEL INVARIANT")
    print("→ D(k) collapse is a global dynamical signature")
    print("→ Very strong result - publishable")
elif collapse_rate >= 0.3:
    print("\n→ Case B: MIXED (depends on trajectory)")
    print("→ D(k) collapse is local to some trajectories")
    print("→ Weaker but meaningful")
else:
    print("\n→ Case C: LOW COLLAPSE RATE")
    print("→ Something wrong in pipeline or regime characterization")

df.to_csv("/home/student/sgp-tribe3/empirical_analysis/neural_networks/basin_results.csv", index=False)
print(f"\nSaved: basin_results.csv")