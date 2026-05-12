import numpy as np
import json
import os

np.random.seed(42)

OUTPUT_DIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase13_operator_path"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading activations...")
X = np.load('/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase10c_training_dynamics/activations.npy')

if len(X.shape) != 2:
    raise ValueError(f"Expected 2D array, got {len(X.shape)}D")
if X.shape[0] < 1000 or X.shape[1] < 10:
    raise ValueError(f"Expected (>=1000, >=10), got {X.shape}")

print(f"Loaded: {X.shape}")

epsilon = 0.1 * np.std(X)
print(f"Epsilon: {epsilon:.4f}")

def identity(x):
    return x

def tanh_1(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def softplus(x):
    return np.log1p(np.exp(x))

depth_1 = [
    ["identity"],
    ["tanh_1"],
    ["relu"],
    ["softplus"]
]

depth_2 = [
    ["tanh_1", "relu"],
    ["relu", "tanh_1"],
    ["softplus", "tanh_1"],
    ["tanh_1", "softplus"]
]

depth_3 = [
    ["tanh_1", "relu", "tanh_1"],
    ["relu", "tanh_1", "relu"],
    ["softplus", "tanh_1", "relu"],
    ["tanh_1", "softplus", "relu"]
]

depth_4 = [
    ["tanh_1", "relu", "tanh_1", "relu"],
    ["relu", "tanh_1", "relu", "tanh_1"],
    ["softplus", "tanh_1", "relu", "tanh_1"],
    ["tanh_1", "softplus", "relu", "tanh_1"]
]

ALL_PATHS = depth_1 + depth_2 + depth_3 + depth_4

op_map = {
    "identity": identity,
    "tanh_1": tanh_1,
    "relu": relu,
    "softplus": softplus
}

def apply_path(X, path):
    X_out = X.copy()
    for op_name in path:
        X_out = op_map[op_name](X_out)
    return X_out

def compute_recurrence(X, eps):
    N = X.shape[0]
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D[i, j] = np.linalg.norm(X[i] - X[j])
    R = (D < eps).astype(np.uint8)
    return R

def compute_F_DET(R):
    N = R.shape[0]
    F = np.sum(R) / (N * N)
    
    diag_counts = 0
    total_counts = np.sum(R)
    l_min = 2
    
    for k in range(-N+1, N):
        diag = np.diagonal(R, offset=k)
        length = 0
        for val in diag:
            if val == 1:
                length += 1
            else:
                if length >= l_min:
                    diag_counts += length
                length = 0
        if length >= l_min:
            diag_counts += length
    
    DET = diag_counts / total_counts if total_counts > 0 else 0
    return F, DET

print("\nComputing baseline...")
R_base = compute_recurrence(X, epsilon)
F_base, DET_base = compute_F_DET(R_base)
print(f"  Baseline F: {F_base:.6f}, DET: {DET_base:.4f}")

print("\nProcessing paths...")
results = []

for path in ALL_PATHS:
    print(f"  Path: {path}")
    X_transformed = apply_path(X, path)
    R = compute_recurrence(X_transformed, epsilon)
    F, DET = compute_F_DET(R)
    delta_F = F - F_base
    delta_DET = DET - DET_base
    depth = len(path)
    
    results.append({
        "path": path,
        "depth": depth,
        "F": float(F),
        "DET": float(DET),
        "delta_F": float(delta_F),
        "delta_DET": float(delta_DET)
    })

scaling = {}

for d in [1, 2, 3, 4]:
    subset = [r for r in results if r["depth"] == d]
    mean_abs_delta_F = np.mean([abs(r["delta_F"]) for r in subset])
    mean_abs_delta_DET = np.mean([abs(r["delta_DET"]) for r in subset])
    scaling[str(d)] = {
        "mean_abs_delta_F": float(mean_abs_delta_F),
        "mean_abs_delta_DET": float(mean_abs_delta_DET)
    }
    print(f"  Depth {d}: mean_abs_delta_F={mean_abs_delta_F:.6f}, mean_abs_delta_DET={mean_abs_delta_DET:.4f}")

for r in results:
    if r["F"] < 0 or r["F"] > 1:
        raise ValueError(f"Invalid F: {r['F']}")
    if r["DET"] < 0 or r["DET"] > 1:
        raise ValueError(f"Invalid DET: {r['DET']}")

output = {
    "baseline": {
        "F": float(F_base),
        "DET": float(DET_base)
    },
    "paths": results,
    "scaling": scaling
}

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved results.json")
print("\nPHASE 13 COMPLETE")