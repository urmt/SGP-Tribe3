# PHASE 8C: INVARIANT DYNAMICAL OBSERVABLE
# STRICT - DO NOT MODIFY ANY PARAMETERS

import numpy as np
import json
import os

np.random.seed(42)

OUTPUT_DIR = "empirical_analysis/physics/phase8c_invariant_observable"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fixed parameters
T = 5000
DISCARD = 1000
DT = 0.01
EPS = 1e-8

# ============================================
# SYSTEM DEFINITIONS
# ============================================

def logistic_step(x, r=3.9):
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
    return np.array([x + dx * DT, y + dy * DT, z + dz * DT])

# ============================================
# DYNAMICAL OBSERVABLE
# ============================================

def compute_invariant_observable(step_func, x0, dim):
    # Initialize trajectories
    x = np.array(x0)
    x_pert = np.array(x0) + EPS
    
    # Store log deltas
    log_deltas = []
    norms = []
    
    # Transient discard
    for _ in range(DISCARD):
        x = step_func(x)
        x_pert = step_func(x_pert)
    
    # Main evolution
    for _ in range(T):
        x = step_func(x)
        x_pert = step_func(x_pert)
        
        # Compute delta
        if dim == 1:
            delta = abs(x_pert - x)
        else:
            delta = np.linalg.norm(x_pert - x)
        
        delta = max(delta, 1e-12)
        log_deltas.append(np.log(delta))
        
        # Compute norm
        if dim == 1:
            norm = abs(x)
        else:
            norm = np.linalg.norm(x)
        norm = max(norm, 1e-12)
        norms.append(norm)
    
    log_deltas = np.array(log_deltas)
    norms = np.array(norms)
    
    # Compute local growth
    local_growth = np.diff(log_deltas)
    
    # Φ_dyn (raw)
    phi_dyn = np.mean(local_growth)
    
    # Φ_inv (normalized by state norm)
    local_growth_normalized = local_growth / norms[:-1]
    phi_inv = np.mean(local_growth_normalized)
    
    # Φ_inv_norm (dimension normalized)
    phi_inv_norm = phi_inv / dim
    
    return {
        "phi_dyn": float(phi_dyn),
        "phi_inv": float(phi_inv),
        "phi_inv_norm": float(phi_inv_norm)
    }

# ============================================
# RUN SYSTEMS
# ============================================

results = {}

# Harmonic (2D)
res = compute_invariant_observable(harmonic_step, np.array([1.0, 0.0]), dim=2)
results["harmonic"] = {
    "lambda_true": 0.0,
    **res
}

# Logistic (1D)
res = compute_invariant_observable(logistic_step, 0.2, dim=1)
results["logistic"] = {
    "lambda_true": 0.49,
    **res
}

# Lorenz (3D)
res = compute_invariant_observable(lorenz_step, np.array([1.0, 1.0, 1.0]), dim=3)
results["lorenz"] = {
    "lambda_true": 0.905,
    **res
}

# ============================================
# VALIDATION
# ============================================

print("Phase 8C Results:")
print("=" * 50)

ordering_holds = True

for name, data in results.items():
    print(f"\n{name}:")
    print(f"  λ_true: {data['lambda_true']}")
    print(f"  Φ_dyn: {data['phi_dyn']:.4f}")
    print(f"  Φ_inv: {data['phi_inv']:.6f}")
    print(f"  Φ_inv_norm: {data['phi_inv_norm']:.6f}")

# Check ordering
phi_vals = [results["harmonic"]["phi_inv"], results["logistic"]["phi_inv"], results["lorenz"]["phi_inv"]]
if not (phi_vals[0] < phi_vals[1] < phi_vals[2]):
    ordering_holds = False
    print("\n⚠️ WARNING: Ordering failed!")
    print(f"  harmonic: {phi_vals[0]:.6f}")
    print(f"  logistic: {phi_vals[1]:.6f}")
    print(f"  lorenz: {phi_vals[2]:.6f}")
else:
    print("\n✅ Ordering holds: harmonic < logistic < lorenz")

# ============================================
# SAVE
# ============================================

output_path = os.path.join(OUTPUT_DIR, "results.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to: {output_path}")