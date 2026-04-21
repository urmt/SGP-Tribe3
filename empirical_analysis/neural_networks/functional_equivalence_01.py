#!/usr/bin/env python3
"""
SFH-SGP_FUNCTIONAL_EQUIVALENCE_01
Test whether Φ depends only on the resulting operator function
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

np.random.seed(42)

# ==================== PART 1: SYSTEM ====================
n = 1000

def logistic_chaotic(n):
    x = 0.5
    xs = []
    for _ in range(n):
        x = 3.9 * x * (1 - x)
        xs.append(x)
    return np.array(xs)

X = logistic_chaotic(n)
X = (X - np.mean(X)) / (np.std(X) + 1e-10)

# ==================== PART 2: OPERATOR PAIRS ====================

# Define operators as functions
def O1_scalar_distributivity(x):
    return 2 * (0.5 * x)

def O2_scalar_distributivity(x):
    return x

def O1_normalize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-10)

def O2_normalize(x):
    return (x - np.mean(x)) * (1 / (np.std(x) + 1e-10))

def O1_redundant_add(x):
    return x + 0

def O2_redundant_add(x):
    return x * 1

def O1_double_norm(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-10)

def O2_double_norm(x):
    y = (x - np.mean(x)) / (np.std(x) + 1e-10)
    return (y - np.mean(y)) / (np.std(y) + 1e-10)

def O1_compose_tanh(x):
    return np.tanh(2 * x)

def O2_compose_tanh(x):
    return 2 * np.tanh(x)

# ==================== PART 3-5: COMPUTE ====================

def recurrence_matrix(x, eps):
    x1 = x.reshape(-1, 1)
    D = pairwise_distances(x1)
    return (D < eps).astype(int)

def compute_alpha(x):
    eps_list = [0.01, 0.05, 0.1]
    rates = []
    for eps in eps_list:
        R = recurrence_matrix(x, eps)
        rates.append(np.sum(R) / (len(x)**2))
    log_eps = np.log(np.array(eps_list) + 1e-10)
    log_r = np.log(np.array(rates) + 1e-10)
    valid = np.isfinite(log_r)
    if valid.sum() >= 2:
        return np.polyfit(log_eps[valid], log_r[valid], 1)[0]
    return 0.0

def compute_det(x, eps=0.1):
    R = recurrence_matrix(x, eps)
    total = np.sum(R)
    if total == 0:
        return 0.0
    diag_sum = 0
    for k in range(-len(R)+1, len(R)):
        d = np.diagonal(R, k)
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

# Define pairs
pairs = [
    ("A1_scalar", "Scalar distributivity", O1_scalar_distributivity, O2_scalar_distributivity),
    ("A2_norm", "Normalization order", O1_normalize, O2_normalize),
    ("B_redundant", "Redundant operations", O1_redundant_add, O2_redundant_add),
    ("C_double_norm", "Double normalization", O1_double_norm, O2_double_norm),
    ("D_compose", "Composition reorder (approx)", O1_compose_tanh, O2_compose_tanh),
]

results = []

print("Testing functional equivalence...")
print("="*60)

for pair_id, pair_name, O1, O2 in pairs:
    # Apply operators
    y1 = O1(X)
    y2 = O2(X)
    
    # Functional difference
    delta_func = np.mean(np.abs(y1 - y2))
    
    # Compute Φ
    alpha1 = compute_alpha(y1)
    det1 = compute_det(y1)
    alpha2 = compute_alpha(y2)
    det2 = compute_det(y2)
    
    # Distance in Φ-space
    d_phi = np.sqrt((alpha1 - alpha2)**2 + (det1 - det2)**2)
    
    results.append({
        "pair_id": pair_id,
        "description": pair_name,
        "delta_func": delta_func,
        "alpha1": alpha1,
        "det1": det1,
        "alpha2": alpha2,
        "det2": det2,
        "d_phi": d_phi
    })
    
    print(f"\n{pair_id} - {pair_name}:")
    print(f"  Δ_functional: {delta_func:.10f}")
    print(f"  Φ(O1): α={alpha1:.4f}, DET={det1:.4f}")
    print(f"  Φ(O2): α={alpha2:.4f}, DET={det2:.4f}")
    print(f"  ΔΦ: {d_phi:.10f}")

df = pd.DataFrame(results)

# ==================== PART 6-7: ANALYSIS ====================
print("\n" + "="*60)
print("RESULTS TABLE")
print("="*60)
print(df[["pair_id", "description", "delta_func", "d_phi"]].to_string(index=False))

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

# Check exact equivalences
exact = df[df["delta_func"] < 1e-10]
approx = df[(df["delta_func"] > 1e-10) & (df["delta_func"] < 0.1)]

print(f"\nExact functional equivalence (Δ_func < 1e-10): {len(exact)} pairs")
if len(exact) > 0:
    print(f"  For these, max ΔΦ: {exact['d_phi'].max():.10f}")
    print(f"  Mean ΔΦ: {exact['d_phi'].mean():.10f}")
    
    if exact['d_phi'].max() < 1e-10:
        print("\n→ YES: Φ depends ONLY on the operator function")
        print("→ Implementation path does NOT matter")
    else:
        print("\n→ PARTIAL: Some exact equivalents give different Φ")

print(f"\nApproximate functional equivalence: {len(approx)} pairs")

# Check if Δ_func ≈ 0 gives ΔΦ ≈ 0
exact_zero = df[df["delta_func"] < 1e-10]
if len(exact_zero) > 0:
    phi_nonzero = (exact_zero["d_phi"] > 1e-10).sum()
    if phi_nonzero == 0:
        conclusion = "INVARIANT - Φ depends only on operator output"
    else:
        conclusion = f"VARIANT - {phi_nonzero}/{len(exact_zero)} exact equivalents give different Φ"
else:
    conclusion = "NO EXACT EQUIVALENTS FOUND"

print(f"\nFinal: {conclusion}")

df.to_csv("/home/student/sgp-tribe3/empirical_analysis/neural_networks/functional_equivalence.csv", index=False)
print("\nSaved: functional_equivalence.csv")