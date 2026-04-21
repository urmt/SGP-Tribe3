#!/usr/bin/env python3
"""
SFH-SGP_TORUS_TEST_01
Test D(k) collapse in quasi-periodic torus attractor
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

n_total = 10000
embedding_dim = 10
delay = 2
k_list = [2, 4, 8, 16]

t = np.linspace(0, 200, n_total)
x = np.sin(t) + np.sin(np.sqrt(2) * t)

if np.isnan(x).any():
    raise ValueError("FAIL: NaN in signal")

def embed(x, dim, delay):
    N = len(x) - (dim - 1) * delay
    return np.array([x[i:i + dim*delay:delay] for i in range(N)])

X = embed(x, embedding_dim, delay)

print(f"Embedded shape: {X.shape}")

if np.std(X) < 1e-10:
    raise ValueError("FAIL: constant signal")

def compute_dk(X, k):
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances = distances[:, 1:] + 1e-10
    return np.mean(np.log(distances))

Dk_values = []

for k in k_list:
    dk = compute_dk(X, k)
    Dk_values.append(dk)

std_Dk = np.std(Dk_values)

print("\nD(k) RESULTS")
print("-" * 30)
for k, val in zip(k_list, Dk_values):
    print(f"k={k}: {val:.4f}")

print(f"\nstd(Dk): {std_Dk:.6f}")

collapse = std_Dk < 0.05

print(f"\nCollapse detected: {collapse}")

df = pd.DataFrame({
    "k": k_list,
    "Dk": Dk_values
})
df["std_Dk"] = std_Dk

output_path = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/torus_results.csv"
df.to_csv(output_path, index=False)

print(f"\nSaved to: {output_path}")

print("\n" + "=" * 50)
print("INTERPRETATION")
print("=" * 50)

if collapse:
    print("\n→ Collapse = TRUE")
    print("→ D(k) detects low-dimensional manifold structure")
    print("   (periodic OR quasi-periodic)")
else:
    print("\n→ Collapse = FALSE")
    print("→ D(k) specifically detects discrete orbit closure")
    print("   (periodicity only, not manifolds)")