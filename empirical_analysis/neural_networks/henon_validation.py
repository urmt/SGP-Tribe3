#!/usr/bin/env python3
"""
SFH-SGP_HENON_VALIDATION_01
Decisive test: Does D(k) collapse occur in Hénon map (chaotic, fractal attractor)?
"""
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/"

np.random.seed(42)

A_VALUES = np.linspace(1.2, 1.4, 40)
B = 0.3

N_TOTAL = 10000
N_BURN = 3000

EMBED_DIM = 10
DELAY = 2
K_LIST = [2, 4, 8, 16]

os.makedirs(OUT_DIR, exist_ok=True)


def henon_map(a, b, n_total, x0=0.1, y0=0.1):
    x, y = x0, y0
    xs = []

    for i in range(n_total):
        x_new = 1 - a * x**2 + y
        y_new = b * x
        x, y = x_new, y_new
        xs.append(x)

    return np.array(xs)


def time_delay_embed(x, dim, delay):
    N = len(x) - (dim - 1) * delay
    embedded = np.zeros((N, dim))
    for i in range(dim):
        embedded[:, i] = x[i*delay : i*delay + N]
    return embedded


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


def lyapunov_henon(a, x):
    x = x[:-1000]
    lambda_est = np.mean(np.log(np.abs(-2 * a * x) + 1e-10))
    return lambda_est


print("=" * 80)
print("SFH-SGP_HENON_VALIDATION_01")
print("=" * 80)

print(f"\nParameter sweep: a = {A_VALUES[0]:.2f} to {A_VALUES[-1]:.2f} ({len(A_VALUES)} points)")
print(f"N_total: {N_TOTAL}, N_burn: {N_BURN}")
print(f"Embedding: dim={EMBED_DIM}, delay={DELAY}")
print(f"K values: {K_LIST}")

results = []

print("\n" + "=" * 80)
print("STEP 1-4: COMPUTE D(k) FOR EACH a")
print("=" * 80)

for idx, a in enumerate(A_VALUES):
    x = henon_map(a, B, N_TOTAL)
    x = x[N_BURN:]
    
    if len(x) < 5000:
        print(f"FAIL: insufficient points at a={a}")
        continue
    
    x = (x - x.mean()) / (x.std() + 1e-10)
    
    embedded = time_delay_embed(x, EMBED_DIM, DELAY)
    
    dk_vals = []
    for k in K_LIST:
        dk = compute_knn_dim(embedded, k)
        dk_vals.append(dk)
    
    std_dk = np.std(dk_vals)
    collapse = std_dk < 0.05
    
    lam = lyapunov_henon(a, x)
    
    results.append({
        "a": a,
        "lambda": lam,
        "D2": dk_vals[0],
        "D4": dk_vals[1],
        "D8": dk_vals[2],
        "D16": dk_vals[3],
        "std_Dk": std_dk,
        "collapse": collapse
    })
    
    if idx < 5 or idx >= len(A_VALUES) - 3:
        print(f"a={a:.3f}: λ={lam:.4f}, std_Dk={std_dk:.4f}, collapse={collapse}")

df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("STEP 5-6: RAW OUTPUT TABLE")
print("=" * 80)

print(f"\n{'a':<8} | {'lambda':<10} | {'D2':<10} | {'D4':<10} | {'D8':<10} | {'D16':<10} | {'std_Dk':<10} | {'collapse':<8}")
print("-" * 95)

for _, row in df.iterrows():
    print(f"{row['a']:<8.3f} | {row['lambda']:<10.4f} | {row['D2']:<10.4f} | {row['D4']:<10.4f} | {row['D8']:<10.4f} | {row['D16']:<10.4f} | {row['std_Dk']:<10.4f} | {row['collapse']}")

print("\n" + "=" * 80)
print("STEP 7: SUMMARY METRICS")
print("=" * 80)

num_chaotic = np.sum(df["lambda"] > 0)
num_collapse = np.sum(df["std_Dk"] < 0.05)
num_overlap = np.sum((df["lambda"] > 0) & (df["std_Dk"] < 0.05))

collapse_rate = num_collapse / len(df)
chaos_rate = num_chaotic / len(df)
overlap_rate = num_overlap / max(num_chaotic, 1)

print(f"\nTotal points: {len(df)}")
print(f"Chaotic points (λ > 0): {num_chaotic}")
print(f"Collapse points: {num_collapse}")
print(f"Overlap (chaos & collapse): {num_overlap}")
print(f"Collapse rate: {collapse_rate:.3f}")
print(f"Chaos rate: {chaos_rate:.3f}")
print(f"Overlap rate (within chaos): {overlap_rate:.3f}")

print("\n" + "=" * 80)
print("STEP 8: SANITY CHECKS")
print("=" * 80)

dk_variation = df[["D2", "D4", "D8", "D16"]].std(axis=1).mean()
nan_count = df.isna().sum().sum()

print(f"\nMean std across D(k): {dk_variation:.4f}")
print(f"NaN count: {nan_count}")
print(f"λ min: {df['lambda'].min():.4f}")
print(f"λ max: {df['lambda'].max():.4f}")

if nan_count > 0:
    raise ValueError("FAIL: NaNs detected")

if dk_variation < 1e-6:
    raise ValueError("FAIL: D(k) is constant (invalid)")

print("\n✅ Sanity checks passed")

print("\n" + "=" * 80)
print("STEP 9: SAVE OUTPUT")
print("=" * 80)

output_path = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/henon_results.csv"
df.to_csv(output_path, index=False)

print(f"Results saved to: {output_path}")

print("\n" + "=" * 80)
print("STEP 10: INTERPRETATION")
print("=" * 80)

print("\nINTERPRETATION")
print("-" * 40)

if num_overlap > 0:
    print("⚠️ Collapse detected inside chaotic regime.")
    print("→ Weakens 'periodicity-only' hypothesis.")
else:
    print("✅ No collapse in chaotic regime.")
    print("→ Supports attractor-geometry hypothesis (collapse ≠ chaos).")

if num_collapse == 0:
    print("✅ No collapse anywhere in Hénon.")
    print("→ Strong evidence collapse is NOT a generic chaos feature.")

print("\n" + "=" * 80)
print("EXACT VALUES REPORT")
print("=" * 80)

print(f"\n1. Collapse count: {num_collapse}")
print(f"2. Chaotic count (λ > 0): {num_chaotic}")
print(f"3. Overlap count: {num_overlap}")
print(f"4. Collapse rate: {collapse_rate:.3f}")
print(f"5. Overlap rate: {overlap_rate:.3f}")
print(f"6. λ min: {df['lambda'].min():.4f}")
print(f"7. λ max: {df['lambda'].max():.4f}")

print("\nSample rows:")
print(df.head(10).to_string())