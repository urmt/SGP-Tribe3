import numpy as np
import json
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

INPUT_FILE = "input_multisystem.json"
OUTPUT_FILE = "results.json"

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples")

for i, d in enumerate(data):
    for key in ["system", "F", "C", "lambda"]:
        if key not in d:
            raise ValueError(f"Missing {key} in sample {i}")

systems = np.array([d["system"] for d in data])
F_raw = np.array([d["F"] for d in data], dtype=float)
C_raw = np.array([d["C"] for d in data], dtype=float)
L = np.array([d["lambda"] for d in data], dtype=float)

has_extra = all("velocity" in d for d in data)
if has_extra:
    V = np.array([d["velocity"] for d in data], dtype=float)
    D = np.array([d["divergence"] for d in data], dtype=float)
else:
    V = None
    D = None

F_mean, F_std = np.mean(F_raw), np.std(F_raw)
C_mean, C_std = np.mean(C_raw), np.std(C_raw)

F_norm = (F_raw - F_mean) / (F_std + 1e-8)
C_norm = (C_raw - C_mean) / (C_std + 1e-8)

train_mask = (systems == "logistic")
test_lorenz = (systems == "lorenz")
test_harmonic = (systems == "harmonic")

X_base = np.column_stack([F_norm, C_norm])

X_nl = np.column_stack([
    F_norm,
    C_norm,
    F_norm * C_norm,
    np.log(np.abs(F_norm) + 1e-6)
])

if V is not None:
    X_hybrid = np.column_stack([F_norm, C_norm, V, D])
else:
    X_hybrid = None

def train_and_eval(X_train, y_train, X_test, y_test, name):
    model = LinearRegression().fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    r_train, _ = pearsonr(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    
    try:
        r_test, _ = pearsonr(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)
    except:
        r_test = float('nan')
        r2_test = float('nan')
    
    return {
        "train_r": float(r_train),
        "train_r2": float(r2_train),
        "test_r": float(r_test),
        "test_r2": float(r2_test)
    }

results = {}

X_train = X_base[train_mask]
y_train = L[train_mask]

X_test_lorenz = X_base[test_lorenz]
y_test_lorenz = L[test_lorenz]

X_test_harmonic = X_base[test_harmonic]
y_test_harmonic = L[test_harmonic]

results["linear_base"] = train_and_eval(X_train, y_train, np.vstack([X_test_lorenz, X_test_harmonic]), np.concatenate([y_test_lorenz, y_test_harmonic]), "linear")

X_nl_train = X_nl[train_mask]
X_nl_test = X_nl[test_lorenz]
X_nl_test_h = X_nl[test_harmonic]

model_nl = LinearRegression().fit(X_nl_train, y_train)
y_pred_train = model_nl.predict(X_nl_train)
y_pred_test = model_nl.predict(np.vstack([X_nl_test, X_nl_test_h]))

r_train, _ = pearsonr(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)
r_test, _ = pearsonr(np.concatenate([y_test_lorenz, y_test_harmonic]), y_pred_test)
r2_test = r2_score(np.concatenate([y_test_lorenz, y_test_harmonic]), y_pred_test)

results["nonlinear"] = {
    "train_r": float(r_train),
    "train_r2": float(r2_train),
    "test_r": float(r_test),
    "test_r2": float(r2_test)
}

model_ridge = Ridge(alpha=1.0).fit(X_train, y_train)
y_pred_train = model_ridge.predict(X_train)
y_pred_test = model_ridge.predict(np.vstack([X_test_lorenz, X_test_harmonic]))

r_train, _ = pearsonr(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)
r_test, _ = pearsonr(np.concatenate([y_test_lorenz, y_test_harmonic]), y_pred_test)
r2_test = r2_score(np.concatenate([y_test_lorenz, y_test_harmonic]), y_pred_test)

results["ridge"] = {
    "train_r": float(r_train),
    "train_r2": float(r2_train),
    "test_r": float(r_test),
    "test_r2": float(r2_test)
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print("\n=== PHASE 9F RESULTS ===")
for name, res in results.items():
    print(f"\n{name}:")
    print(f"  Train: r={res['train_r']:.3f}, R²={res['train_r2']:.3f}")
    print(f"  Test:  r={res['test_r']:.3f}, R²={res['test_r2']:.3f}")

if all(r['test_r2'] < 0 for r in results.values()):
    print("\n❌ Cross-system generalization FAILED")

if any(r['train_r2'] - r['test_r2'] > 0.5 for r in results.values()):
    print("\n⚠️ Model is system-specific (overfit to dynamics)")