#!/usr/bin/env python3
"""
SFH-SGP_NOISY_PERIODIC_01
Test noise robustness of D(k) collapse detection
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

sigma_list = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2]

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

print(f"Base signal (sigma=0): {len(x)} points")
print(f"Embedding: {embedding_dim}D, delay={delay}")

for sigma in sigma_list:
    
    x_noisy = x + sigma * np.random.randn(len(x))
    
    if np.isnan(x_noisy).any():
        raise ValueError("NaN detected")
    
    X = embed(x_noisy, embedding_dim, delay)
    
    Dk = []
    for k in k_list:
        Dk.append(compute_dk(X, k))
    
    std_Dk = np.std(Dk)
    collapse = std_Dk < 0.05
    
    results.append({
        "sigma": sigma,
        "D2": Dk[0],
        "D4": Dk[1],
        "D8": Dk[2],
        "D16": Dk[3],
        "std_Dk": std_Dk,
        "collapse": collapse
    })
    
    print(f"sigma={sigma:.3f}: std_Dk={std_Dk:.4f} collapse={collapse}")

df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("NOISE ROBUSTNESS RESULTS")
print("=" * 60)
print(df.to_string(index=False))

df.to_csv("/home/student/sgp-tribe3/empirical_analysis/neural_networks/noise_results.csv", index=False)

# Find where collapse breaks
collapse_states = df["collapse"].values
sigma_values = df["sigma"].values

first_non_collapse = None
for i, (s, c) in enumerate(zip(sigma_values, collapse_states)):
    if not c:
        first_non_collapse = s
        break

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

if first_non_collapse:
    print(f"\n→ Collapse breaks at sigma = {first_non_collapse}")
    
    if first_non_collapse <= 0.01:
        print("→ FRAGILE: Only detects exact periodicity")
    elif first_non_collapse <= 0.1:
        print("→ MODERATE: Detects approximate periodicity")
    else:
        print("→ ROBUST: Detects noisy repetition")
else:
    print("\n→ Collapse survives all noise levels")
    print("→ ROBUST: Detects noisy repetition")