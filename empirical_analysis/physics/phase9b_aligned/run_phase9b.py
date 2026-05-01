import numpy as np
import json

r_values = np.linspace(3.5, 4.0, 25)
operators = ["identity", "tanh"]

T = 2000
T_transient = 500

def logistic(x, r):
    return r * x * (1 - x)

def apply_operator(x, op):
    if op == "identity":
        return x
    elif op == "tanh":
        return np.tanh(x)
    else:
        raise ValueError("Unknown operator")

def compute_lambda(r, x_series):
    vals = []
    for x in x_series:
        vals.append(np.log(abs(r * (1 - 2*x)) + 1e-12))
    return np.mean(vals)

def compute_recurrence_metrics(x):
    eps = 0.05
    N = len(x)
    R = np.abs(x[:, None] - x[None, :]) < eps
    DET = np.mean(R)
    F = DET
    var = np.var([np.mean(R[i]) for i in range(N)])
    C = 1.0 / (var + 1e-8)
    return F, C

results = []

for op in operators:
    for r in r_values:
        x = 0.5
        trajectory = []
        for t in range(T):
            x = logistic(x, r)
            if t > T_transient:
                trajectory.append(x)
        trajectory = np.array(trajectory)
        lam = compute_lambda(r, trajectory)
        obs = apply_operator(trajectory, op)
        F, C = compute_recurrence_metrics(obs)
        results.append({
            "r": float(r),
            "operator": op,
            "F": float(F),
            "C": float(C),
            "lambda": float(lam)
        })

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Dataset built:", len(results), "samples")