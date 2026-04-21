#!/usr/bin/env python3
"""
SFH-SGP_ENSEMBLE_COLLAPSE_01
Statistical generalization: segment-based collapse probability
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

r_list = [3.5, 3.57, 3.9]
sigma_list = [0.0, 1e-4, 1e-3]

n_total = 8000
n_burn = 2000

segment_length = 300
n_segments = 50

embedding_dim = 10
delay = 2
k_list = [2,4,8,16]

collapse_threshold = 0.05

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

print("Running ensemble collapse experiment...")

for r in r_list:
    for sigma in sigma_list:
        
        x0 = np.random.rand()
        x = logistic(r, x0, n_total)[n_burn:]
        x = x + sigma * np.random.randn(len(x))
        
        collapse_count = 0
        std_Dk_values = []
        
        for i in range(n_segments):
            
            start = np.random.randint(0, len(x) - segment_length)
            seg = x[start:start + segment_length]
            
            X = embed(seg, embedding_dim, delay)
            
            Dk = [compute_dk(X, k) for k in k_list]
            std_Dk = np.std(Dk)
            std_Dk_values.append(std_Dk)
            
            if std_Dk < collapse_threshold:
                collapse_count += 1
        
        collapse_prob = collapse_count / n_segments
        
        results.append({
            "r": r,
            "sigma": sigma,
            "collapse_prob": collapse_prob,
            "mean_std_Dk": np.mean(std_Dk_values),
            "std_std_Dk": np.std(std_Dk_values)
        })

df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("ENSEMBLE COLLAPSE RESULTS")
print("=" * 60)
print(df.to_string(index=False))

print("\n" + "=" * 60)
print("ANALYSIS BY REGIME")
print("=" * 60)

for r in r_list:
    subset = df[df["r"] == r]
    print(f"\nr = {r}:")
    for _, row in subset.iterrows():
        print(f"  σ={row['sigma']:.0e}: collapse_prob={row['collapse_prob']:.3f}")

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

periodic = df[df["r"] == 3.5]
chaotic = df[df["r"] == 3.9]

if periodic["collapse_prob"].mean() > 0.5 and chaotic["collapse_prob"].mean() < 0.2:
    print("\n→ Ensemble method SEPARATES periodic from chaotic")
    print("→ Can work as noise-robust detector")
elif periodic["collapse_prob"].mean() > 0.3:
    print("\n→ Partial separation")
    print("→ May need tuning")
else:
    print("\n→ No clear separation")
    print("→ Need redesign")

df.to_csv("/home/student/sgp-tribe3/empirical_analysis/neural_networks/ensemble_collapse_results.csv", index=False)
print(f"\nSaved: ensemble_collapse_results.csv")