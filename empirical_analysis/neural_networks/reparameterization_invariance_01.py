#!/usr/bin/env python3
"""
SFH-SGP_REPARAMETRIZATION_INVARIANCE_01
Test whether Φ(S, O) is invariant under reparameterization
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

np.random.seed(42)

# ==================== PART 1: SYSTEM ====================
n = 1500

def logistic_chaotic(n):
    x = 0.5
    xs = []
    for _ in range(n):
        x = 3.9 * x * (1 - x)
        xs.append(x)
    return np.array(xs)

X = logistic_chaotic(n)
X = (X - np.mean(X)) / (np.std(X) + 1e-10)

# ==================== PART 2-3: TWO PARAMETERIZATIONS ====================
n_pts = 40

# Original parameterization: λ ∈ [0,1]
lam_vals = np.linspace(0, 1, n_pts)

# Reparameterized: μ = λ^2, so λ = sqrt(μ)
# Same operators, different parameter path
mu_vals = lam_vals ** 2  # Square the parameter

def apply_operator(x, param, type="original"):
    if type == "original":
        lam = param
        return lam * x + (1 - lam) * np.sin(x)
    else:  # reparameterized
        mu = param
        lam = np.sqrt(mu)
        return lam * x + (1 - lam) * np.sin(x)

# ==================== PART 4: COMPUTE Φ ====================
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

print("Computing Φ for both parameterizations...")

results_original = []
results_reparam = []

for lam, mu in zip(lam_vals, mu_vals):
    # Original: λ parameterization
    obs_orig = apply_operator(X, lam, "original")
    alpha_orig = compute_alpha(obs_orig)
    det_orig = compute_det(obs_orig)
    results_original.append({
        "param": lam,
        "mu_equiv": mu,
        "alpha": alpha_orig,
        "DET": det_orig
    })
    
    # Reparameterized: μ = λ^2 parameterization
    obs_reparam = apply_operator(X, mu, "reparam")
    alpha_reparam = compute_alpha(obs_reparam)
    det_reparam = compute_det(obs_reparam)
    results_reparam.append({
        "param": mu,
        "alpha": alpha_reparam,
        "DET": det_reparam
    })

df_orig = pd.DataFrame(results_original)
df_reparam = pd.DataFrame(results_reparam)

# ==================== PART 5-6: ALIGN AND COMPARE ====================
print("\n" + "="*60)
print("REPARAMETERIZATION INVARIANCE TEST")
print("="*60)

# Match by equivalent operator (λ = sqrt(μ))
aligned = []
for i in range(len(lam_vals)):
    lam = lam_vals[i]
    mu = mu_vals[i]
    # Same operator: λ = sqrt(μ)
    obs1 = apply_operator(X, lam, "original")
    obs2 = apply_operator(X, mu, "reparam")
    
    alpha1 = compute_alpha(obs1)
    det1 = compute_det(obs1)
    alpha2 = compute_alpha(obs2)
    det2 = compute_det(obs2)
    
    d_phi = np.sqrt((alpha1 - alpha2)**2 + (det1 - det2)**2)
    
    aligned.append({
        "lambda": lam,
        "mu": mu,
        "alpha_orig": alpha1,
        "det_orig": det1,
        "alpha_reparam": alpha2,
        "det_reparam": det2,
        "d_phi": d_phi
    })

df_align = pd.DataFrame(aligned)

print("\nComparison of same operators:")
print(df_align[["lambda", "mu", "alpha_orig", "alpha_reparam", "d_phi"]].head(10).to_string())

# ==================== PART 7: STATISTICS ====================
print("\n" + "="*60)
print("RESULTS")
print("="*60)

mean_diff = df_align["d_phi"].mean()
max_diff = df_align["d_phi"].max()
std_diff = df_align["d_phi"].std()

print(f"\nMean ΔΦ: {mean_diff:.6f}")
print(f"Max ΔΦ: {max_diff:.6f}")
print(f"Std ΔΦ: {std_diff:.6f}")

# ==================== PART 8: INTERPRETATION ====================
print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

print("\n1) Is Φ representation-invariant?")
if mean_diff < 0.01 and max_diff < 0.05:
    print(f"   YES (mean ΔΦ = {mean_diff:.6f}) → Φ depends only on operator, not parameterization")
elif mean_diff < 0.1:
    print(f"   PARTIAL (mean ΔΦ = {mean_diff:.6f}) → small dependence on parameterization")
else:
    print(f"   NO (mean ΔΦ = {mean_diff:.6f}) → Φ depends on parameterization")

print("\n2) Where do differences occur?")
print(f"   α difference: mean = {np.abs(df_align['alpha_orig'] - df_align['alpha_reparam']).mean():.6f}")
print(f"   DET difference: mean = {np.abs(df_align['det_orig'] - df_align['det_reparam']).mean():.6f}")

df_align.to_csv("/home/student/sgp-tribe3/empirical_analysis/neural_networks/reparameterization_test.csv", index=False)
print("\nSaved: reparameterization_test.csv")