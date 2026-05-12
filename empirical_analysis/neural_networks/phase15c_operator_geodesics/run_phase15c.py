import os
import json
import numpy as np

BASE_PATH = "/home/student/sgp-tribe3/empirical_analysis/neural_networks"
PHASE10C_PATH = os.path.join(BASE_PATH, "phase10c_training_dynamics")
OUTPUT_PATH = os.path.join(BASE_PATH, "phase15c_operator_geodesics")

ACTIVATIONS_FILE = os.path.join(PHASE10C_PATH, "activations.npy")

if not os.path.exists(ACTIVATIONS_FILE):
    raise FileNotFoundError("ERROR: activations.npy missing")

os.makedirs(OUTPUT_PATH, exist_ok=True)

X = np.load(ACTIVATIONS_FILE)

if X.shape[0] < 300:
    raise ValueError("ERROR: insufficient samples")

X = X[:300]

def identity(x): return x
def tanh(x): return np.tanh(x)
def relu(x): return np.maximum(0, x)
def softplus(x): return np.log1p(np.exp(x))
def sigmoid(x): return 1 / (1 + np.exp(-x))

OPS = {
    "identity": identity,
    "tanh": tanh,
    "relu": relu,
    "softplus": softplus,
    "sigmoid": sigmoid
}

names = list(OPS.keys())

def normalize_activations(x):
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    valid = std > 1e-8
    if np.sum(valid) < x.shape[1] * 0.5:
        raise RuntimeError("ERROR: too many collapsed dimensions")
    x = x[:, valid[0]]
    mean = mean[:, valid[0]]
    std = std[:, valid[0]]
    x = (x - mean) / std
    
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise RuntimeError("ERROR: zero vector detected")
    x = x / norms
    
    return x

def recurrence_fixed(x):
    eps = 0.1
    dist = np.linalg.norm(x[:, None] - x[None, :], axis=2)
    return (dist < eps).astype(int)

def recurrence_variable(x, eps):
    dist = np.linalg.norm(x[:, None] - x[None, :], axis=2)
    return (dist < eps).astype(int)

def F_metric(R):
    return np.sum(R) / (R.shape[0] ** 2)

def DET_metric(R):
    N = R.shape[0]
    diag_lengths = []

    for k in range(-N+1, N):
        diag = np.diag(R, k=k)
        length = 0
        for v in diag:
            if v == 1:
                length += 1
            else:
                if length >= 2:
                    diag_lengths.append(length)
                length = 0
        if length >= 2:
            diag_lengths.append(length)

    if len(diag_lengths) == 0:
        return 0.0

    return np.sum(diag_lengths) / np.sum(R)

def alpha_metric(x):
    x = normalize_activations(x)
    eps_vals = np.logspace(-4, -1, 10)
    
    if len(eps_vals) != 10:
        raise RuntimeError("ERROR: epsilon count modified")
    
    if not np.isclose(eps_vals[0], 1e-4) or not np.isclose(eps_vals[-1], 1e-1):
        raise RuntimeError("ERROR: epsilon range modified")
    
    rates = []

    for eps in eps_vals:
        R = recurrence_variable(x, eps)
        rates.append(np.sum(R) / (R.shape[0] ** 2))

    rates = np.array(rates)
    coeffs = np.polyfit(np.log(eps_vals), np.log(rates + 1e-8), 1)
    return coeffs[0]

EXPECTED_EPSILON_RECUR = 0.1
EXPECTED_EPS_RANGE = (1e-4, 1e-1)
EXPECTED_EPS_COUNT = 10

def validate_epsilon():
    test_vals = np.logspace(-4, -1, 10)
    if len(test_vals) != EXPECTED_EPS_COUNT:
        raise RuntimeError("ERROR: epsilon count modified")
    if not np.isclose(test_vals[0], EXPECTED_EPS_RANGE[0]):
        raise RuntimeError("ERROR: epsilon min modified")
    if not np.isclose(test_vals[-1], EXPECTED_EPS_RANGE[1]):
        raise RuntimeError("ERROR: epsilon max modified")

validate_epsilon()

def Phi(x):
    x = normalize_activations(x)
    R = recurrence_fixed(x)
    return np.array([
        F_metric(R),
        DET_metric(R),
        alpha_metric(x)
    ])

phi_base = []
for name in names:
    phi_base.append(Phi(OPS[name](X)))

phi_base = np.array(phi_base)

mean = np.mean(phi_base, axis=0)
std = np.std(phi_base, axis=0)

if np.any(std == 0):
    raise RuntimeError("ERROR: zero std")

def normalize(v):
    return (v - mean) / std

results = []

for A in names:
    for B in names:
        if A == B:
            continue

        X_direct = OPS[A](OPS[B](X))
        phi_direct = normalize(Phi(X_direct))

        for C in names:
            if C == A or C == B:
                continue

            X_path = OPS[A](OPS[C](OPS[B](X)))
            phi_path = normalize(Phi(X_path))

            deviation = np.linalg.norm(phi_direct - phi_path)

            results.append({
                "A": A,
                "C": C,
                "B": B,
                "deviation": float(deviation)
            })

deviations = [r["deviation"] for r in results]

summary = {
    "mean_deviation": float(np.mean(deviations)),
    "max_deviation": float(np.max(deviations)),
    "min_deviation": float(np.min(deviations)),
    "num_paths": len(results)
}

top = sorted(results, key=lambda x: -x["deviation"])[:20]

if len(results) == 0:
    raise RuntimeError("ERROR: no paths computed")

if np.isnan(deviations).any():
    raise RuntimeError("ERROR: NaN in deviations")

output = {
    "summary": summary,
    "top_deviations": top,
    "all_results": results[:100]
}

with open(os.path.join(OUTPUT_PATH, "results.json"), "w") as f:
    json.dump(output, f, indent=4)

print("\nPHASE 15C COMPLETE — PATH GEOMETRY ANALYZED")
print(f"Paths tested: {summary['num_paths']}")
print(f"Mean deviation: {summary['mean_deviation']:.6f}")
print(f"Max deviation: {summary['max_deviation']:.6f}")