# PHASE 9A (FAST): DYNAMICS → OBSERVABILITY

import numpy as np
import json
import os
from scipy.integrate import trapezoid

np.random.seed(42)

OUTPUT_DIR = 'empirical_analysis/physics/phase9a_coupling_fast'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# PARAMETERS
r_values = np.linspace(3.5, 4.0, 25)
T_total = 6000
T_transient = 1000
eps_values = np.logspace(-4, -1, 15)
subsample = 5
window = 100
l_min = 2

# OPERATORS
operators = {
    "identity": lambda x: x,
    "tanh": lambda x: np.tanh(x),
    "relu": lambda x: np.maximum(0, x),
    "softplus": lambda x: np.log1p(np.exp(x))
}

# FAST DET COMPUTATION
def compute_det_fast(x, eps):
    N = len(x)
    lines = 0
    total = 0
    
    for i in range(N - window):
        for j in range(i+1, i+window):
            if j >= N:
                break
            if abs(x[i] - x[j]) < eps:
                length = 1
                while (i+length < N and j+length < N and abs(x[i+length] - x[j+length]) < eps):
                    length += 1
                if length >= l_min:
                    lines += length
                total += length
    
    if total == 0:
        return 0.0
    
    return lines / total

# MAIN LOOP
results = []

for r in r_values:
    print(f"r={r:.2f}", end=" ")
    
    # trajectory
    x = 0.5
    traj = []
    for t in range(T_total):
        x = r * x * (1 - x)
        if t > T_transient:
            traj.append(x)
    
    traj = np.array(traj)[::subsample]
    
    # true lambda
    log_sum = sum(np.log(abs(r*(1-2*v))+1e-12) for v in traj)
    lambda_true = log_sum / len(traj)
    
    # operators
    for name, op in operators.items():
        y = op(traj)
        
        DET_vals = []
        for eps in eps_values:
            DET = compute_det_fast(y, eps)
            DET_vals.append(DET)
        
        DET_vals = np.array(DET_vals)
        
        F = trapezoid(DET_vals, eps_values)
        var = np.var(DET_vals)
        C = 1.0 / (var + 1e-12)
        
        results.append({
            "r": float(r),
            "lambda": float(lambda_true),
            "operator": name,
            "F": float(F),
            "C": float(C)
        })

print("\nComputing correlations...")

# Organize by operator
by_op = {}
for r in results:
    op = r["operator"]
    if op not in by_op:
        by_op[op] = {"lambda": [], "F": [], "C": []}
    by_op[op]["lambda"].append(r["lambda"])
    by_op[op]["F"].append(r["F"])
    by_op[op]["C"].append(r["C"])

# Correlations
correlations = {}
for op, data in by_op.items():
    lam = np.array(data["lambda"])
    F = np.array(data["F"])
    C = np.array(data["C"])
    correlations[op] = {
        "r_F_lambda": float(np.corrcoef(F, lam)[0,1]),
        "r_C_lambda": float(np.corrcoef(C, lam)[0,1])
    }

print("\nCorrelations:")
for op, c in correlations.items():
    print(f"  {op}: r(F,l)={c['r_F_lambda']:.3f}, r(C,l)={c['r_C_lambda']:.3f}")

# Save
output = {"results": results, "correlations": correlations}
with open(f"{OUTPUT_DIR}/results.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to {OUTPUT_DIR}/results.json")