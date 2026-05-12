import numpy as np
import json
import os

np.random.seed(42)

OUTPUT_DIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase12_operator_composition"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading activations...")
X = np.load('/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase10c_training_dynamics/activations.npy')
print(f"Loaded: {X.shape}")

if X.shape != (2000, 64):
    raise ValueError(f"Expected (2000, 64), got {X.shape}")

epsilon = 0.5
l_min = 2

def O_identity(x):
    return x

def O_tanh(x):
    return np.tanh(x)

def O_relu(x):
    return np.maximum(0, x)

def O_softplus(x):
    return np.log1p(np.exp(x))

def O_tanh_relu(x):
    return O_tanh(O_relu(x))

def O_relu_tanh(x):
    return O_relu(O_tanh(x))

def O_softplus_tanh(x):
    return O_softplus(O_tanh(x))

def O_tanh_softplus(x):
    return O_tanh(O_softplus(x))

def compute_recurrence(X, eps):
    N = X.shape[0]
    R = np.zeros((N, N), dtype=np.uint8)
    for i in range(N):
        for j in range(N):
            dist = np.linalg.norm(X[i] - X[j])
            if dist < eps:
                R[i, j] = 1
    return R

def compute_F(R):
    return np.sum(R) / (R.shape[0] * R.shape[1])

def compute_DET(R, l_min):
    N = R.shape[0]
    diag_counts = 0
    total_counts = np.sum(R)
    
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
    
    if total_counts == 0:
        return 0
    
    return diag_counts / total_counts

base_operators = {
    "identity": O_identity,
    "tanh": O_tanh,
    "relu": O_relu,
    "softplus": O_softplus
}

composed_operators = {
    "tanh_relu": O_tanh_relu,
    "relu_tanh": O_relu_tanh,
    "softplus_tanh": O_softplus_tanh,
    "tanh_softplus": O_tanh_softplus
}

base_results = {}
composed_results = {}

print("\nRunning base operators...")
for name, op in base_operators.items():
    print(f"  {name}...")
    X_op = op(X)
    R = compute_recurrence(X_op, epsilon)
    F = compute_F(R)
    DET = compute_DET(R, l_min)
    base_results[name] = {"F": float(F), "DET": float(DET)}
    print(f"    F={F:.4f}, DET={DET:.4f}")

print("\nRunning composed operators...")
for name, op in composed_operators.items():
    print(f"  {name}...")
    X_op = op(X)
    R = compute_recurrence(X_op, epsilon)
    F = compute_F(R)
    DET = compute_DET(R, l_min)
    composed_results[name] = {"F": float(F), "DET": float(DET)}
    print(f"    F={F:.4f}, DET={DET:.4f}")

additivity_test = {}

compositions = [
    ("tanh_relu", "tanh", "relu"),
    ("relu_tanh", "relu", "tanh"),
    ("softplus_tanh", "softplus", "tanh"),
    ("tanh_softplus", "tanh", "softplus")
]

for comp_name, op1_name, op2_name in compositions:
    F_comp = composed_results[comp_name]["F"]
    F_op1 = base_results[op1_name]["F"]
    F_op2 = base_results[op2_name]["F"]
    delta_F = F_comp - (F_op1 + F_op2)
    
    DET_comp = composed_results[comp_name]["DET"]
    DET_op1 = base_results[op1_name]["DET"]
    DET_op2 = base_results[op2_name]["DET"]
    delta_DET = DET_comp - (DET_op1 + DET_op2)
    
    additivity_test[comp_name] = {
        "delta_F": float(delta_F),
        "delta_DET": float(delta_DET)
    }
    print(f"\n{comp_name}:")
    print(f"  delta_F = {delta_F:.4f}")
    print(f"  delta_DET = {delta_DET:.4f}")

for name in base_results:
    F = base_results[name]["F"]
    DET = base_results[name]["DET"]
    if F < 0 or F > 1 or DET < 0 or DET > 1:
        raise ValueError(f"Invalid value in base {name}")

for name in composed_results:
    F = composed_results[name]["F"]
    DET = composed_results[name]["DET"]
    if F < 0 or F > 1 or DET < 0 or DET > 1:
        raise ValueError(f"Invalid value in composed {name}")

all_delta_F_zero = all(abs(additivity_test[k]["delta_F"]) < 1e-6 for k in additivity_test)
if all_delta_F_zero:
    print("\nWARNING: additive behavior (unexpected)")

results = {
    "base": base_results,
    "composed": composed_results,
    "additivity_test": additivity_test
}

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved results.json")
print("\nPHASE 12 COMPLETE")