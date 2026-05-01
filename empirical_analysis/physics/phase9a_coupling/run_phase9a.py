# PHASE 9A: DYNAMICS → OBSERVABILITY COUPLING

import numpy as np
import json
import os
from scipy.integrate import trapezoid

np.random.seed(42)

OUTPUT_DIR = "empirical_analysis/physics/phase9a_coupling"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# PARAMETERS
# ============================================

r_values = np.linspace(3.5, 4.0, 25)
T_total = 5000
T_transient = 1000
eps_values = np.logspace(-4, -1, 20)
l_min = 2

# ============================================
# OPERATORS
# ============================================

operators = {
    "identity": lambda x: x,
    "tanh": lambda x: np.tanh(x),
    "relu": lambda x: np.maximum(0, x),
    "softplus": lambda x: np.log(1 + np.exp(x))
}

# ============================================
# RECURRENCE FUNCTIONS
# ============================================

def compute_recurrence_matrix(x, eps):
    n = len(x)
    R = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            if abs(x[i] - x[j]) < eps:
                R[i, j] = 1
    return R

def compute_DET(R, l_min):
    n = R.shape[0]
    total = 0
    for k in range(-n+1, n):
        diag = np.diag(R, k=k)
        length = 0
        for val in diag:
            if val == 1:
                length += 1
            else:
                if length >= l_min:
                    total += length
                length = 0
        if length >= l_min:
            total += length
    return total / (np.sum(R) + 1e-12)

# ============================================
# MAIN LOOP
# ============================================

results = []

for r in r_values:
    print(f"Computing r = {r:.3f}...")
    
    # Generate trajectory
    x = 0.5
    traj = []
    for t in range(T_total):
        x = r * x * (1 - x)
        if t > T_transient:
            traj.append(x)
    traj = np.array(traj)
    
    # True Lyapunov (ground truth)
    log_sum = 0.0
    for val in traj:
        d = abs(r * (1 - 2 * val))
        if d < 1e-12:
            d = 1e-12
        log_sum += np.log(d)
    lambda_true = log_sum / len(traj)
    
    # Apply operators
    for op_name, op in operators.items():
        y = op(traj)
        
        # Recurrence analysis
        DET_values = []
        for eps in eps_values:
            R = compute_recurrence_matrix(y, eps)
            DET = compute_DET(R, l_min)
            DET_values.append(DET)
        
        DET_values = np.array(DET_values)
        
        # Metrics
        F = trapezoid(DET_values, eps_values)
        variance = np.var(DET_values)
        C = 1.0 / (variance + 1e-12)
        
        results.append({
            "r": float(r),
            "lambda": float(lambda_true),
            "operator": op_name,
            "F": float(F),
            "C": float(C)
        })

# ============================================
# ANALYSIS
# ============================================

print("\nAnalyzing correlations...")

# Organize by operator
by_operator = {}
for r in results:
    op = r["operator"]
    if op not in by_operator:
        by_operator[op] = {"lambda": [], "F": [], "C": []}
    by_operator[op]["lambda"].append(r["lambda"])
    by_operator[op]["F"].append(r["F"])
    by_operator[op]["C"].append(r["C"])

# Compute correlations
correlations = {}

for op, data in by_operator.items():
    lam = np.array(data["lambda"])
    F = np.array(data["F"])
    C = np.array(data["C"])
    
    correlations[op] = {
        "r_F_lambda": float(np.corrcoef(F, lam)[0, 1]),
        "r_C_lambda": float(np.corrcoef(C, lam)[0, 1]),
        "r_C_F": float(np.corrcoef(C, F)[0, 1])
    }

# Print summary
print("\n" + "=" * 50)
print("CORRELATIONS BY OPERATOR")
print("=" * 50)

for op, corr in correlations.items():
    print(f"\n{op}:")
    print(f"  r(F, λ):   {corr['r_F_lambda']:.4f}")
    print(f"  r(C, λ):   {corr['r_C_lambda']:.4f}")
    print(f"  r(C, F):   {corr['r_C_F']:.4f}")

# Save full results
output = {
    "results": results,
    "correlations": correlations
}

output_path = os.path.join(OUTPUT_DIR, "results.json")
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to: {output_path}")