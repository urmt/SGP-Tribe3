# PHASE 8D: STABILIZED DYNAMICAL OBSERVABLE
# (TRUE LYAPUNOV-STYLE ESTIMATION)
# STRICT - DO NOT MODIFY PARAMETERS

import numpy as np
import json
import os

np.random.seed(42)

OUTPUT_DIR = "empirical_analysis/physics/phase8d_stable_dynamics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fixed parameters
T_total = 6000
T_transient = 1000
epsilon = 1e-8
DT = 0.01

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
# STABILIZED LYAPUNOV ESTIMATION
# ============================================

def compute_lyapunov_stable(step_func, x0):
    # Initialize state
    x = np.array(x0)
    
    # Initialize perturbation (random small vector)
    v = np.random.randn(*x.shape)
    v = v / np.linalg.norm(v) * epsilon
    
    log_sum = 0.0
    count = 0
    
    for t in range(T_total):
        # Evolve base trajectory
        x_next = step_func(x)
        
        # Evolve perturbed trajectory
        x_perturbed = x + v
        x_perturbed_next = step_func(x_perturbed)
        
        # Compute new deviation vector
        v_new = x_perturbed_next - x_next
        
        v_norm = np.linalg.norm(v_new)
        
        # Skip if too small (avoid log(0))
        if v_norm > 1e-12:
            g = v_norm / np.linalg.norm(v)
            log_sum += np.log(g)
            count += 1
        
        # Renormalize perturbation
        if v_norm > 1e-12:
            v = (v_new / v_norm) * epsilon
        else:
            # Reset to random if collapsed
            v = np.random.randn(*x.shape)
            v = v / np.linalg.norm(v) * epsilon
        
        # Update state
        x = x_next
    
    # Compute final Lyapunov estimate (after transient)
    if count > 0:
        phi_dyn_stable = log_sum / count
    else:
        phi_dyn_stable = 0.0
    
    return phi_dyn_stable

# ============================================
# RUN SYSTEMS
# ============================================

results = {}

# Harmonic (2D)
phi = compute_lyapunov_stable(harmonic_step, np.array([1.0, 0.0]))
results["harmonic"] = {
    "lambda_true": 0.0,
    "phi_dyn_stable": float(phi)
}

# Logistic (1D)
phi = compute_lyapunov_stable(logistic_step, np.array([0.2]))
results["logistic"] = {
    "lambda_true": 0.49,
    "phi_dyn_stable": float(phi)
}

# Lorenz (3D)
phi = compute_lyapunov_stable(lorenz_step, np.array([1.0, 1.0, 1.0]))
results["lorenz"] = {
    "lambda_true": 0.905,
    "phi_dyn_stable": float(phi)
}

# ============================================
# VALIDATION
# ============================================

print("Phase 8D Results:")
print("=" * 50)

ordering_holds = True

for name, data in results.items():
    print(f"\n{name}:")
    print(f"  λ_true:   {data['lambda_true']}")
    print(f"  Φ_dyn:    {data['phi_dyn_stable']:.4f}")

# Check ordering
phi_vals = [results["harmonic"]["phi_dyn_stable"], 
            results["logistic"]["phi_dyn_stable"], 
            results["lorenz"]["phi_dyn_stable"]]

if phi_vals[0] < phi_vals[1] < phi_vals[2]:
    print("\n✅ Ordering holds: harmonic < logistic < lorenz")
else:
    print("\n⚠️ WARNING: Ordering failed!")
    print(f"  harmonic: {phi_vals[0]:.6f}")
    print(f"  logistic: {phi_vals[1]:.6f}")
    print(f"  lorenz:   {phi_vals[2]:.6f}")

# ============================================
# SAVE
# ============================================

output_path = os.path.join(OUTPUT_DIR, "results.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to: {output_path}")