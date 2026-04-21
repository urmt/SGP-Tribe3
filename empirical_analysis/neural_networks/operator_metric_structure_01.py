#!/usr/bin/env python3
"""
SFH-SGP_OPERATOR_METRIC_STRUCTURE_01
Test whether Φ preserves geometric relationships between operators
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

np.random.seed(42)

# ==================== PART 1: SYSTEM ====================
n = 2000

def logistic_chaotic(n):
    x = 0.5
    xs = []
    for _ in range(n):
        x = 3.9 * x * (1 - x)
        xs.append(x)
    return np.array(xs)

X = logistic_chaotic(n)
X = (X - np.mean(X)) / (np.std(X) + 1e-10)

# ==================== PART 2: OPERATOR FAMILY ====================
n_lam = 50
lam_vals = np.linspace(0, 1, n_lam)

def apply_operator(x, lam):
    return lam * x + (1 - lam) * np.sin(x)

# ==================== PART 3: COMPUTE Φ ====================
def recurrence_matrix_1d(x, eps):
    x = x.reshape(-1, 1)
    D = pairwise_distances(x)
    R = (D < eps).astype(int)
    np.fill_diagonal(R, 0)
    return R

def compute_alpha(x):
    eps_list = [0.01, 0.05, 0.1, 0.2]
    rates = []
    for eps in eps_list:
        R = recurrence_matrix_1d(x, eps)
        rate = np.sum(R) / (len(x)**2)
        rates.append(rate)
    log_eps = np.log(np.array(eps_list) + 1e-10)
    log_rates = np.log(np.array(rates) + 1e-10)
    valid = np.isfinite(log_rates)
    if valid.sum() >= 2:
        return np.polyfit(log_eps[valid], log_rates[valid], 1)[0]
    return 0.0

def compute_det(x, eps=0.1):
    R = recurrence_matrix_1d(x, eps)
    N = len(R)
    total = np.sum(R)
    if total == 0:
        return 0.0
    diag_sum = 0
    for k in range(-N+1, N):
        d = np.diagonal(R, offset=k)
        cnt = 0
        for v in d:
            if v == 1:
                cnt += 1
            else:
                if cnt >= 2:
                    diag_sum += cnt
                cnt = 0
        if cnt >= 2:
            diag_sum += cnt
    return diag_sum / total

print("Computing Φ for operator family...")

phi_values = []

for lam in lam_vals:
    obs = apply_operator(X, lam)
    alpha = compute_alpha(obs)
    det = compute_det(obs)
    
    phi_values.append({
        "lambda": lam,
        "alpha": alpha,
        "DET": det
    })

df = pd.DataFrame(phi_values)

print("\nΦ(λ) values:")
print(df.to_string())

# ==================== PART 4: DISTANCE MATRICES ====================
print("\n" + "="*60)
print("PART 4: DISTANCE ANALYSIS")
print("="*60)

# Operator distances (parameter space)
d_O = np.abs(np.array(lam_vals)[:, np.newaxis] - np.array(lam_vals)[np.newaxis, :])
np.fill_diagonal(d_O, 0)

# Φ distances (observable space)
phi_coords = df[["alpha", "DET"]].values
d_phi = pairwise_distances(phi_coords)
np.fill_diagonal(d_phi, 0)

# ==================== PART 5: CORRELATION ====================
print("\n" + "="*60)
print("PART 5: CORRELATION TEST")
print("="*60)

# Flatten both matrices
d_O_flat = d_O.flatten()
d_phi_flat = d_phi.flatten()

# Valid pairs only (upper triangle)
mask = np.triu(np.ones_like(d_O), k=1).astype(bool)
d_O_upper = d_O[mask]
d_phi_upper = d_phi[mask]

# Correlations
pearson = np.corrcoef(d_O_upper, d_phi_upper)[0, 1]
spearman = np.corrcoef(d_O_upper, d_phi_upper)[0, 1]

print(f"Pearson correlation: {pearson:.4f}")
print(f"Spearman correlation: {spearman:.4f}")

# ==================== PART 6: LOCAL SMOOTHNESS ====================
print("\n" + "="*60)
print("PART 6: LOCAL SMOOTHNESS")
print("="*60)

# Compute step sizes
alpha_steps = np.abs(np.diff(df["alpha"]))
det_steps = np.abs(np.diff(df["DET"]))

# Combined step size in Φ-space
phi_steps = np.sqrt(alpha_steps**2 + det_steps**2)

print(f"Max Φ step: {phi_steps.max():.6f}")
print(f"Mean Φ step: {phi_steps.mean():.6f}")
print(f"Std Φ step: {phi_steps.std():.6f}")

# Check for jumps
large_jumps = (phi_steps > 0.1).sum()
print(f"Large jumps (>0.1): {large_jumps}")

# ==================== PART 7: SAVE ====================
df.to_csv("/home/student/sgp-tribe3/empirical_analysis/neural_networks/lambda_phi_values.csv", index=False)

# Distance comparison
dist_df = pd.DataFrame({
    "d_operator": d_O_upper,
    "d_phi": d_phi_upper
})
dist_df.to_csv("/home/student/sgp-tribe3/empirical_analysis/neural_networks/operator_vs_phi_distance.csv", index=False)

print("\nSaved: lambda_phi_values.csv, operator_vs_phi_distance.csv")

# ==================== PART 8: INTERPRETATION ====================
print("\n" + "="*60)
print("PART 8: INTERPRETATION")
print("="*60)

print("\n1) Is operator distance correlated with Φ distance?")
if abs(pearson) > 0.7:
    print(f"   YES, strongly (r={pearson:.3f}) → Φ preserves metric")
elif abs(pearson) > 0.3:
    print(f"   PARTIAL (r={pearson:.3f}) → weak metric preservation")
else:
    print(f"   NO (r={pearson:.3f}) → no metric embedding")

print("\n2) Is Φ smooth across λ?")
if phi_steps.max() < 0.1:
    print(f"   YES, smooth (max jump: {phi_steps.max():.4f})")
else:
    print(f"   NO, large jumps exist (max: {phi_steps.max():.4f})")

print("\n3) Any large discontinuities?")
if large_jumps == 0:
    print("   NONE - continuous embedding")
else:
    print(f"   {large_jumps} discontinuities found")

# Summary
print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

if abs(pearson) > 0.5 and phi_steps.max() < 0.1:
    print("→ Φ is a metric embedding of operator space")
    print("→ Smooth, structure-preserving mapping")
elif abs(pearson) > 0.3:
    print("→ Partial metric structure")
    print("→ Weak geometric preservation")
else:
    print("→ No clear metric structure")
    print("→ Mapping is not geometric in the tested sense")