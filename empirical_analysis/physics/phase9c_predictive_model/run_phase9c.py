import numpy as np
import json
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

DATA_FILE = "../phase7b_lyapunov/phase7b_results.json"
OUTPUT_FILE = "results.json"

with open(DATA_FILE, 'r') as f:
    raw = json.load(f)

data = raw["data"]

F = np.array([d["F"] for d in data])
C = np.array([d["C"] for d in data])
L = np.array([d["lyapunov"] for d in data])

print(f"Data points: {len(F)}")

X1 = np.column_stack([F, C])
model1 = LinearRegression().fit(X1, L)
L_pred1 = model1.predict(X1)
r1, _ = pearsonr(L, L_pred1)
r2_1 = r2_score(L, L_pred1)

F_safe = np.clip(F, 1e-8, None)
X2 = np.column_stack([np.log(F_safe), C])
model2 = LinearRegression().fit(X2, L)
L_pred2 = model2.predict(X2)
r2, _ = pearsonr(L, L_pred2)
r2_2 = r2_score(L, L_pred2)

X3 = np.column_stack([F, C, F*C])
model3 = LinearRegression().fit(X3, L)
L_pred3 = model3.predict(X3)
r3, _ = pearsonr(L, L_pred3)
r2_3 = r2_score(L, L_pred3)

results = {
    "model_1_linear": {
        "coefficients": model1.coef_.tolist(),
        "intercept": float(model1.intercept_),
        "pearson_r": float(r1),
        "r2": float(r2_1)
    },
    "model_2_log": {
        "coefficients": model2.coef_.tolist(),
        "intercept": float(model2.intercept_),
        "pearson_r": float(r2),
        "r2": float(r2_2)
    },
    "model_3_interaction": {
        "coefficients": model3.coef_.tolist(),
        "intercept": float(model3.intercept_),
        "pearson_r": float(r3),
        "r2": float(r2_3)
    }
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print("\n=== PHASE 9C RESULTS ===")
for name, res in results.items():
    print(f"\n{name}: r={res['pearson_r']:.3f}, R^2={res['r2']:.3f}")