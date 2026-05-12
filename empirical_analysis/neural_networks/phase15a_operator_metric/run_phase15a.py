import os
import json
import numpy as np

BASE_PATH = "/home/student/sgp-tribe3/empirical_analysis/neural_networks"
PHASE10C_PATH = os.path.join(BASE_PATH, "phase10c_training_dynamics")
OUTPUT_PATH = os.path.join(BASE_PATH, "phase15a_operator_metric")

ACTIVATIONS_FILE = os.path.join(PHASE10C_PATH, "activations.npy")

if not os.path.exists(ACTIVATIONS_FILE):
    raise FileNotFoundError("ERROR: activations.npy not found. DO NOT REBUILD.")

os.makedirs(OUTPUT_PATH, exist_ok=True)

activations = np.load(ACTIVATIONS_FILE)

if activations.shape[0] < 300:
    raise ValueError("ERROR: Not enough samples")

subset = activations[:300]

def identity(x):
    return x

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def softplus(x):
    return np.log1p(np.exp(x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

OPERATORS = {
    "identity": identity,
    "tanh": tanh,
    "relu": relu,
    "softplus": softplus,
    "sigmoid": sigmoid
}

def compute_recurrence(x, eps=5.0):
    N = x.shape[0]
    dist = np.linalg.norm(x[:, None] - x[None, :], axis=2)
    R = (dist < eps).astype(int)
    return R

def compute_F(R):
    return np.sum(R) / (R.shape[0] ** 2)

def compute_DET(R):
    diag_counts = []
    N = R.shape[0]

    for k in range(-N+1, N):
        diag = np.diag(R, k=k)
        length = 0
        for val in diag:
            if val == 1:
                length += 1
            else:
                if length >= 2:
                    diag_counts.append(length)
                length = 0
        if length >= 2:
            diag_counts.append(length)

    if len(diag_counts) == 0:
        return 0.0

    diag_counts = np.array(diag_counts)
    return np.sum(diag_counts) / np.sum(R)

def compute_alpha(x):
    eps_vals = [1.0, 2.0, 5.0, 10.0, 15.0]
    rec_rates = []

    for eps in eps_vals:
        R = compute_recurrence(x, eps)
        rec_rates.append(np.sum(R) / (R.shape[0] ** 2))

    rec_rates = np.array(rec_rates)
    coeffs = np.polyfit(np.log(eps_vals), np.log(rec_rates + 1e-8), 1)
    return coeffs[0]

phi = {}

for name, op in OPERATORS.items():
    print(f"Processing operator: {name}")

    transformed = op(subset)
    
    transformed = (transformed - np.mean(transformed, axis=1, keepdims=True)) / (np.std(transformed, axis=1, keepdims=True) + 1e-8)

    R = compute_recurrence(transformed)

    F = compute_F(R)
    DET = compute_DET(R)
    alpha = compute_alpha(transformed)

    phi[name] = {
        "F": float(F),
        "DET": float(DET),
        "alpha": float(alpha)
    }

names = list(OPERATORS.keys())
D = np.zeros((len(names), len(names)))

def vectorize(p):
    return np.array([p["F"], p["DET"], p["alpha"]])

for i, ni in enumerate(names):
    for j, nj in enumerate(names):
        vi = vectorize(phi[ni])
        vj = vectorize(phi[nj])
        D[i, j] = np.linalg.norm(vi - vj)

output = {
    "operators": names,
    "phi": phi,
    "distance_matrix": D.tolist()
}

with open(os.path.join(OUTPUT_PATH, "results.json"), "w") as f:
    json.dump(output, f, indent=4)

if len(phi) != 5:
    raise RuntimeError("ERROR: Missing operators")

if D.shape != (5, 5):
    raise RuntimeError("ERROR: Distance matrix incorrect shape")

if np.isnan(D).any():
    raise RuntimeError("ERROR: NaN detected in distance matrix")

print("\nPHASE 15A COMPLETE — OPERATOR METRIC SPACE GENERATED")