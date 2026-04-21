#!/usr/bin/env python3
"""
SFH-SGP_OPERATOR_EQUIVALENCE_01
Test whether operator space ℴ partitions into equivalence classes under Φ(S, O)
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances

np.random.seed(42)

# ==================== PART 1: SYSTEMS ====================
n_points = 1000

def logistic_chaotic(n):
    x = 0.5
    xs = []
    for _ in range(n):
        x = 3.9 * x * (1 - x)
        xs.append(x)
    return np.array(xs)

def periodic_signal(n):
    t = np.linspace(0, 50, n)
    return np.sin(2 * np.pi * t / 10)

def stochastic_signal(n):
    return np.random.randn(n)

systems = {
    "chaotic": logistic_chaotic(n_points),
    "periodic": periodic_signal(n_points),
    "stochastic": stochastic_signal(n_points)
}

# ==================== PART 2: OPERATORS ====================
def apply_operator(x, op_name):
    x = x - np.mean(x)
    x = x / (np.std(x) + 1e-10)
    
    if op_name == "mean":
        return x
    elif op_name == "weighted":
        return x * np.linspace(0.5, 1.5, len(x))
    elif op_name == "random_proj":
        return x * np.random.rand(len(x))
    elif op_name == "sin":
        return np.sin(x)
    elif op_name == "tanh":
        return np.tanh(x * 2)
    elif op_name == "square":
        return x ** 2
    elif op_name == "zscore":
        return (x - np.mean(x)) / (np.std(x) + 1e-10)
    elif op_name == "minmax":
        return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-10)
    elif op_name.startswith("mixed_"):
        lam = float(op_name.split("_")[1])
        return lam * x + (1 - lam) * np.sin(x)
    return x

# Operator list
operators = ["mean", "weighted", "random_proj", "sin", "tanh", "square", "zscore", "minmax"]
for i in range(11):
    operators.append(f"mixed_{i/10:.1f}")

# ==================== PART 3: COMPUTE Φ ====================
def recurrence_matrix_1d(x, eps):
    x = x.reshape(-1, 1)
    D = pairwise_distances(x)
    R = (D < eps).astype(int)
    np.fill_diagonal(R, 0)
    return R

def compute_alpha(x):
    eps_list = [0.01, 0.05, 0.1]
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

def compute_det(x, eps=0.05):
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

results = []

print("Computing Φ(S, O)...")

for sys_name, sys_signal in systems.items():
    print(f"\n{sys_name}:")
    
    for op_name in operators:
        try:
            obs = apply_operator(sys_signal, op_name)
            alpha = compute_alpha(obs)
            det = compute_det(obs)
            
            results.append({
                "system": sys_name,
                "operator": op_name,
                "alpha": alpha,
                "DET": det
            })
            print(f"  {op_name}: α={alpha:.3f}, DET={det:.3f}")
        except Exception as e:
            print(f"  {op_name}: ERROR - {e}")

df = pd.DataFrame(results)

# ==================== PART 4-6: ANALYSIS ====================
print("\n" + "="*60)
print("EQUIVALENCE ANALYSIS")
print("="*60)

epsilon = 0.1

for sys_name in systems.keys():
    subset = df[df["system"] == sys_name].copy()
    
    X = subset[["alpha", "DET"]].values
    
    # Standardize
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    
    # Pairwise distances
    D = pairwise_distances(X_std)
    np.fill_diagonal(D, 0)
    
    # Equivalence pairs
    eq_pairs = np.sum(D < epsilon) // 2
    total_pairs = len(subset) * (len(subset) - 1) // 2
    
    # Clustering
    sil_scores = []
    for k in range(2, min(6, len(subset))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_std)
        sil = silhouette_score(X_std, labels)
        sil_scores.append((k, sil))
    
    best_k = max(sil_scores, key=lambda x: x[1])[0]
    
    print(f"\n{sys_name}:")
    print(f"  Equivalence pairs (ε={epsilon}): {eq_pairs}/{total_pairs} ({100*eq_pairs/max(total_pairs,1):.1f}%)")
    print(f"  Best k={best_k} (silhouette={sil_scores[best_k-2][1]:.3f})")
    
    # Show sample operators and their Φ
    print(f"  Sample Φ values:")
    for i in range(min(5, len(subset))):
        print(f"    {subset.iloc[i]['operator']}: α={subset.iloc[i]['alpha']:.3f}, DET={subset.iloc[i]['DET']:.3f}")

# ==================== SAVE ====================
df.to_csv("/home/student/sgp-tribe3/empirical_analysis/neural_networks/operator_phi_values.csv", index=False)
print("\nSaved: operator_phi_values.csv")

# ==================== INTERPRETATION ====================
print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

# Check for equivalence collapse
for sys_name in systems.keys():
    subset = df[df["system"] == sys_name]
    unique_phi = len(subset[["alpha", "DET"]].drop_duplicates())
    total_ops = len(subset)
    print(f"\n{sys_name}: {unique_phi} unique Φ values from {total_ops} operators")
    if unique_phi < total_ops * 0.5:
        print(f"  → Multiple operators collapse to same Φ (equivalence classes exist)")