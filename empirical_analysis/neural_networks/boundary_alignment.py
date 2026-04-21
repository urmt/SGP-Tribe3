#!/usr/bin/env python3
"""
SFH-SGP_BOUNDARY_ALIGNMENT_09
High-resolution D(k) collapse vs Lyapunov zero-crossing alignment
"""
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/"

np.random.seed(42)

# ============== PARAMETERS ==============
R_MIN, R_MAX, R_STEP = 3.54, 3.60, 0.002
N_TOTAL, N_BURN = 6000, 1000
EMBED_DIM, DELAY = 10, 2
K_LIST = [2, 4, 8, 16]
COLLAPSE_THRESH = 0.05

# ============== FUNCTIONS ==============
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

def lyapunov(r, x0=0.5, n=5000):
    x = x0
    lambdas = []
    for i in range(n):
        x = r * x * (1 - x)
        if i >= 100:
            deriv = abs(r * (1 - 2 * x))
            if deriv > 0:
                lambdas.append(np.log(deriv))
    return np.mean(lambdas) if lambdas else 0

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 80)
print("BOUNDARY ALIGNMENT ANALYSIS")
print("=" * 80)

r_values = np.round(np.arange(R_MIN, R_MAX + R_STEP, R_STEP), 3)
print(f"\nr range: {R_MIN} to {R_MAX}, step={R_STEP}")
print(f"Number of r values: {len(r_values)}")

results = []

for r in r_values:
    x = 0.5
    trajectory = np.zeros(N_TOTAL)
    for i in range(N_TOTAL):
        x = r * x * (1 - x)
        trajectory[i] = x
    
    signal = trajectory[N_BURN:]
    
    if len(signal) < 4000:
        print(f"FAIL: insufficient points at r={r}")
        continue
    
    signal = (signal - signal.mean()) / (signal.std() + 1e-10)
    
    embedded = time_delay_embed(signal, EMBED_DIM, DELAY)
    
    dk_vals = []
    for k in K_LIST:
        dk = compute_knn_dim(embedded, k)
        dk_vals.append(dk)
    
    std_dk = np.std(dk_vals)
    auc = np.trapz(dk_vals, np.log(K_LIST))
    
    lam = lyapunov(r)
    collapse = std_dk < COLLAPSE_THRESH
    
    results.append({
        'r': r,
        'lambda': lam,
        'D2': dk_vals[0],
        'D4': dk_vals[1],
        'D8': dk_vals[2],
        'D16': dk_vals[3],
        'std_Dk': std_dk,
        'AUC': auc,
        'collapse': collapse
    })

df = pd.DataFrame(results).dropna()

print("\n" + "=" * 80)
print("STEP 5: FULL OUTPUT TABLE")
print("=" * 80)

print(f"\n{'r':<8} | {'lambda':<12} | {'std_Dk':<12} | {'collapse':<10} | {'AUC':<12}")
print("-" * 70)
for _, row in df.iterrows():
    col_str = str(row['collapse'])
    print(f"{row['r']:<8.3f} | {row['lambda']:<12.6f} | {row['std_Dk']:<12.6f} | {col_str:<10} | {row['AUC']:<12.6f}")

print("\n" + "=" * 80)
print("STEP 9: SANITY CHECKS")
print("=" * 80)

lambda_arr = df['lambda'].values
std_dk_arr = df['std_Dk'].values
collapse_arr = df['collapse'].values

print(f"\nλ array ({len(lambda_arr)} values):")
print(lambda_arr)

print(f"\nstd_Dk array ({len(std_dk_arr)} values):")
print(std_dk_arr)

print(f"\ncollapse array ({len(collapse_arr)} values):")
print(collapse_arr)

# Check for sign changes in lambda
sign_changes = np.where(np.diff(np.sign(lambda_arr)))[0]
print(f"\nλ sign changes at indices: {sign_changes}")

if len(sign_changes) != 1:
    print(f"\nWARNING: Expected exactly 1 sign change, found {len(sign_changes)}")

# Check collapse is contiguous
collapse_indices = np.where(collapse_arr)[0]
if len(collapse_indices) > 0:
    is_contiguous = np.all(np.diff(collapse_indices) == 1)
    print(f"Collapse region contiguous: {is_contiguous}")
else:
    is_contiguous = False
    print("No collapse detected!")

print("\n" + "=" * 80)
print("STEP 6: BOUNDARY DETECTION")
print("=" * 80)

# Find λ zero crossing
neg_idx = np.where(lambda_arr < 0)[0]
pos_idx = np.where(lambda_arr > 0)[0]

if len(neg_idx) > 0 and len(pos_idx) > 0:
    last_neg = neg_idx[-1]
    first_pos = pos_idx[0]
    
    r1, r2 = df.iloc[last_neg]['r'], df.iloc[first_pos]['r']
    l1, l2 = lambda_arr[last_neg], lambda_arr[first_pos]
    
    r_lambda_zero = r1 + (0 - l1) * (r2 - r1) / (l2 - l1)
    print(f"λ zero crossing (interpolated): r = {r_lambda_zero:.6f}")
else:
    r_lambda_zero = np.nan
    print("Could not find λ zero crossing")

# Find collapse boundary
if collapse_arr.any():
    collapse_end_idx = np.where(collapse_arr)[0][-1]
    r_collapse_end = df.iloc[collapse_end_idx]['r']
    print(f"Collapse region ends at: r = {r_collapse_end:.3f}")
else:
    r_collapse_end = np.nan
    print("No collapse detected")

print("\n" + "=" * 80)
print("STEP 7: ALIGNMENT METRIC")
print("=" * 80)

if not np.isnan(r_lambda_zero) and not np.isnan(r_collapse_end):
    alignment_error = abs(r_lambda_zero - r_collapse_end)
    print(f"\nAlignment error: {alignment_error:.6f}")
else:
    alignment_error = np.nan
    print("\nCannot compute alignment error")

print("\n" + "=" * 80)
print("STEP 8: DECISION CRITERIA")
print("=" * 80)

if not np.isnan(alignment_error):
    if alignment_error < 0.01:
        conclusion = "STRONG ALIGNMENT"
    elif alignment_error < 0.02:
        conclusion = "MODERATE ALIGNMENT"
    else:
        conclusion = "WEAK / FAIL"
    
    print(f"\nConclusion: {conclusion}")
else:
    conclusion = "INSUFFICIENT DATA"
    print(f"\nConclusion: {conclusion}")

print("\n" + "=" * 80)
print("STEP 10: FINAL OUTPUT")
print("=" * 80)

print(f"\nr_lambda_zero: {r_lambda_zero:.6f}" if not np.isnan(r_lambda_zero) else "\nr_lambda_zero: N/A")
print(f"r_collapse_end: {r_collapse_end:.6f}" if not np.isnan(r_collapse_end) else "r_collapse_end: N/A")
print(f"alignment_error: {alignment_error:.6f}" if not np.isnan(alignment_error) else "alignment_error: N/A")

print(f"\nConclusion:")
if not np.isnan(alignment_error) and alignment_error < 0.02:
    print("Does D(k) collapse align with λ = 0 boundary? YES")
else:
    print("Does D(k) collapse align with λ = 0 boundary? NO")

df.to_csv(os.path.join(OUT_DIR, "boundary_alignment.csv"), index=False)
print(f"\nSaved: {OUT_DIR}boundary_alignment.csv")