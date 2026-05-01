# STRICT SCRIPT — DO NOT MODIFY ANY PARAMETERS
# PURPOSE: Compute dynamical observable Φ_dyn and compare to Lyapunov λ_L

import numpy as np
import json
import os

np.random.seed(42)

OUTPUT_DIR = "empirical_analysis/physics/phase8b_dynamical_observable"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPS = 1e-8
T = 3000
DT = 0.01

# -----------------------------
# SYSTEM DEFINITIONS
# -----------------------------

def logistic_map(x, r=3.9):
    return r * x * (1 - x)

def harmonic_step(state):
    x, v = state
    dx = v
    dv = -x
    return np.array([x + dx * DT, v + dv * DT])

def lorenz_step(state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([
        x + dx * DT,
        y + dy * DT,
        z + dz * DT
    ])

# -----------------------------
# Φ_dyn COMPUTATION
# -----------------------------

def compute_phi_dyn_1d(f, x0):
    x = x0
    x_pert = x0 + EPS

    logs = []

    for _ in range(T):
        x_next = f(x)
        x_pert_next = f(x_pert)

        d = abs(x_pert - x) + 1e-12
        d_next = abs(x_pert_next - x_next) + 1e-12

        logs.append(np.log(d_next / d))

        x, x_pert = x_next, x_pert_next

    logs = np.array(logs)

    return {
        "phi_dyn_mean": float(np.mean(logs)),
        "phi_dyn_std": float(np.std(logs))
    }

def compute_phi_dyn_nd(f, x0):
    x = x0
    x_pert = x0 + EPS * np.random.randn(*x0.shape)

    logs = []

    for _ in range(T):
        x_next = f(x)
        x_pert_next = f(x_pert)

        d = np.linalg.norm(x_pert - x) + 1e-12
        d_next = np.linalg.norm(x_pert_next - x_next) + 1e-12

        logs.append(np.log(d_next / d))

        x, x_pert = x_next, x_pert_next

    logs = np.array(logs)

    return {
        "phi_dyn_mean": float(np.mean(logs)),
        "phi_dyn_std": float(np.std(logs))
    }

# -----------------------------
# RUN SYSTEMS
# -----------------------------

results = []

# Logistic
res = compute_phi_dyn_1d(logistic_map, 0.2)
results.append({
    "system": "logistic",
    **res
})

# Harmonic
res = compute_phi_dyn_nd(harmonic_step, np.array([1.0, 0.0]))
results.append({
    "system": "harmonic",
    **res
})

# Lorenz
res = compute_phi_dyn_nd(lorenz_step, np.array([1.0, 1.0, 1.0]))
results.append({
    "system": "lorenz",
    **res
})

# -----------------------------
# SAVE
# -----------------------------

output_path = os.path.join(OUTPUT_DIR, "phase8b_results.json")

with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print("Phase 8B COMPLETE")
print("Results saved to:", output_path)