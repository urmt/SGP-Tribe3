#!/usr/bin/env python3
"""
SFH-SGP_FLAT_REGION_VERIFY_01
Verify whether flat D(k) region is real or artifact
"""
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

EMBED_DIM = 10
DELAY = 2
K_VALUES = [2, 4, 8, 16]
OUT_DIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/"

np.random.seed(42)

def compute_knn_dim(data, k):
    try:
        n, d = data.shape
        if n <= k + 1 or d < 2:
            return np.nan
        data_clean = data.copy()
        if np.std(data_clean) < 1e-10:
            return np.nan
        nn = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree')
        nn.fit(data_clean)
        dists, _ = nn.kneighbors(data_clean)
        return np.mean(np.log(dists[:, -1] + 1e-10))
    except:
        return np.nan

def time_delay_embed(x, dim, delay):
    N = len(x) - (dim - 1) * delay
    embedded = np.zeros((N, dim))
    for i in range(dim):
        embedded[:, i] = x[i*delay : i*delay + N]
    return embedded

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("STEP 1: ZOOM IN TO FLAT REGION")
print("=" * 70)

N_SAMPLES = 5000
TRANSIENT = 1000
R_MIN, R_MAX, R_STEP = 3.40, 3.60, 0.005

r_values = np.round(np.arange(R_MIN, R_MAX + R_STEP, R_STEP), 3)
print(f"\nr range: {R_MIN} to {R_MAX}, step={R_STEP}")
print(f"Points: {len(r_values)}")

print("\n" + "=" * 70)
print("STEP 2-3: RECOMPUTE AND VARIANCE CHECK")
print("=" * 70)

results = []

for r in r_values:
    x = 0.5
    samples = np.zeros(N_SAMPLES + TRANSIENT)
    for i in range(N_SAMPLES + TRANSIENT):
        x = r * x * (1 - x)
        samples[i] = x
    
    signal = samples[TRANSIENT:]
    signal = (signal - signal.mean()) / (signal.std() + 1e-10)
    
    embedded = time_delay_embed(signal, EMBED_DIM, DELAY)
    
    dk = {}
    for k in K_VALUES:
        dk[k] = compute_knn_dim(embedded, k)
    
    results.append({
        'r': r,
        'D2': dk[2],
        'D4': dk[4],
        'D8': dk[8],
        'D16': dk[16]
    })

df = pd.DataFrame(results).dropna()

print(f"\nGenerated: {len(df)} points")

print("\n" + "=" * 70)
print("STEP 4: OUTPUT TABLE")
print("=" * 70)

dk_cols = ['D2', 'D4', 'D8', 'D16']

for col in dk_cols:
    df[col + '_var'] = df[col]

df['dk_std'] = df[dk_cols].std(axis=1)

print(f"\n{'r':<8} {'D2':<10} {'D4':<10} {'D8':<10} {'D16':<10} {'std':<10}")
print("-" * 65)
for _, row in df.iterrows():
    print(f"{row['r']:<8.3f} {row['D2']:<10.4f} {row['D4']:<10.4f} {row['D8']:<10.4f} {row['D16']:<10.4f} {row['dk_std']:<10.4f}")

print("\n" + "=" * 70)
print("VARIANCE ANALYSIS")
print("=" * 70)

std_per_r = df['dk_std'].values
mean_std = np.mean(std_per_r)
max_std = np.max(std_per_r)
min_std = np.min(std_per_r)

print(f"\nVariance stats across r:")
print(f"  Mean std: {mean_std:.8f}")
print(f"  Max std: {max_std:.8f}")
print(f"  Min std: {min_std:.8f}")

is_constant_k = max_std < 1e-10
is_constant_region = mean_std < 1e-10

print("\n" + "=" * 70)
print("FINAL QUESTIONS")
print("=" * 70)

print(f"\n1. EXACTLY constant across k? {'YES' if is_constant_k else 'NO'}")
print(f"   (max std = {max_std:.10f})")

abrupt = (min_std < 1e-6) and (max_std > 1e-6)
gradual = (min_std > 1e-6) and (max_std - min_std) < 1e-6

print(f"2. Abrupt or gradual collapse? {'ABRUPT' if abrupt else ('GRADUAL' if gradual else 'NON-MONOTONIC')}")

after_chaos = df[df['r'] > 3.57]['dk_std'].values
reappears = np.mean(after_chaos) > mean_std * 2 if len(after_chaos) > 0 else False

print(f"3. Variability reappears after chaos? {'YES' if reappears else 'NO'}")

df.to_csv(os.path.join(OUT_DIR, "flat_region_verify.csv"), index=False)
print(f"\nSaved: {OUT_DIR}flat_region_verify.csv")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

if is_constant_k:
    print("\nD(k) IS LITERALLY CONSTANT in this region")
    print("This is likely due to numerical properties, not system behavior")
    print("The embedding may be saturated at fixed points")
else:
    print("\nD(k) shows SMALL but NON-ZERO variance")
    print("This suggests the flat region is NOT numerical artifact")