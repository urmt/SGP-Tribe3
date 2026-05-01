import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

DATA_PATH = "../phase9b_aligned/results.json"

with open(DATA_PATH, "r") as f:
    data = json.load(f)

REQUIRED_KEYS = ["F", "C", "lambda", "operator"]

for i, d in enumerate(data):
    for key in REQUIRED_KEYS:
        if key not in d:
            raise ValueError(f"Missing key '{key}' in entry {i}")

F = np.array([d["F"] for d in data], dtype=float)
C = np.array([d["C"] for d in data], dtype=float)
L = np.array([d["lambda"] for d in data], dtype=float)
O = np.array([d["operator"] for d in data])

assert len(F) > 20, "Insufficient data points"
assert np.all(np.isfinite(F)), "F contains NaN/Inf"
assert np.all(np.isfinite(C)), "C contains NaN/Inf"
assert np.all(np.isfinite(L)), "λ contains NaN/Inf"

operators = np.unique(O)

results = {}

for op in operators:
    idx = (O == op)
    F_op = F[idx]
    C_op = C[idx]
    L_op = L[idx]
    
    if len(F_op) < 10:
        print(f"Skipping operator {op} (too few samples)")
        continue
    
    X = np.column_stack([F_op, C_op])
    model = LinearRegression().fit(X, L_op)
    L_pred = model.predict(X)
    r, _ = pearsonr(L_op, L_pred)
    r2 = r2_score(L_op, L_pred)
    
    results[op] = {
        "n_samples": int(len(F_op)),
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "pearson_r": float(r),
        "r2": float(r2)
    }

with open("results_verified.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== PHASE 9C' VERIFIED RESULTS ===")

for op, res in results.items():
    print(f"\nOperator: {op}")
    print(f"Samples: {res['n_samples']}")
    print(f"r = {res['pearson_r']:.3f}")
    print(f"R^2 = {res['r2']:.3f}")