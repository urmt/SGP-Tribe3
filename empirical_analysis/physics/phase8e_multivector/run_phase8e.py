# PHASE 8E: MULTI-VECTOR LYAPUNOV (STRICT)

import numpy as np
import json
import os

np.random.seed(42)

OUTPUT_DIR = "empirical_analysis/physics/phase8e_multivector"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fixed parameters
T_total = 10000
T_transient = 2000
epsilon = 1e-8
dt = 0.01

# ============================================
# SYSTEM DEFINITIONS
# ============================================

def f_logistic(x, r=3.9):
    return r * x * (1 - x)

def f_harmonic(x):
    return np.array([x[1], -x[0]])

def f_lorenz(x, sigma=10, rho=28, beta=8/3):
    dx = sigma * (x[1] - x[0])
    dy = x[0] * (rho - x[2]) - x[1]
    dz = x[0] * x[1] - beta * x[2]
    return np.array([dx, dy, dz])

# ============================================
# RK4 INTEGRATION
# ============================================

def step_ode(f, x):
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ============================================
# CORE LYAPUNOV ROUTINE
# ============================================

def compute_lyapunov(system):
    
    # ====================
    # LOGISTIC (DISCRETE)
    # ====================
    if system == "logistic":
        x = np.array([0.5])
        log_sum = 0.0
        r = 3.9
        
        for t in range(T_total):
            x_next = r * x * (1 - x)
            if t > T_transient:
                deriv = abs(r * (1 - 2 * x[0]))
                if deriv < 1e-12:
                    deriv = 1e-12
                log_sum += np.log(deriv)
            x = x_next
        
        lambdas = np.array([log_sum / (T_total - T_transient)])
        return sorted(lambdas, reverse=True)
    
    # ====================
    # HARMONIC (CONTINUOUS)
    # ====================
    elif system == "harmonic":
        D = 2
        x = np.array([1.0, 0.0])
        
        def step(x): return step_ode(f_harmonic, x)
        
        V = np.eye(D) * epsilon
        log_sums = np.zeros(D)
        
        for t in range(T_total):
            x_next = step(x)
            V_new = np.zeros((D, D))
            
            for i in range(D):
                x_perturbed = x + V[:, i]
                V_new[:, i] = step(x_perturbed) - x_next
            
            Q, R = np.linalg.qr(V_new)
            
            if t > T_transient:
                for i in range(D):
                    val = abs(R[i, i])
                    if val < 1e-16:
                        val = 1e-16
                    log_sums[i] += np.log(val)
            
            V = Q
            x = x_next
        
        lambdas = log_sums / ((T_total - T_transient) * dt)
        return sorted(lambdas, reverse=True)
    
    # ====================
    # LORENZ (CONTINUOUS)
    # ====================
    elif system == "lorenz":
        D = 3
        x = np.array([1.0, 1.0, 1.0])
        
        def step(x): return step_ode(f_lorenz, x)
        
        V = np.eye(D) * epsilon
        log_sums = np.zeros(D)
        
        for t in range(T_total):
            x_next = step(x)
            V_new = np.zeros((D, D))
            
            for i in range(D):
                x_perturbed = x + V[:, i]
                V_new[:, i] = step(x_perturbed) - x_next
            
            Q, R = np.linalg.qr(V_new)
            
            if t > T_transient:
                for i in range(D):
                    val = abs(R[i, i])
                    if val < 1e-16:
                        val = 1e-16
                    log_sums[i] += np.log(val)
            
            V = Q
            x = x_next
        
        lambdas = log_sums / ((T_total - T_transient) * dt)
        return sorted(lambdas, reverse=True)
    
    else:
        raise ValueError(f"Unknown system: {system}")

# ============================================
# RUN SYSTEMS
# ============================================

results = {}

for system in ["logistic", "harmonic", "lorenz"]:
    print(f"Computing {system}...")
    lambdas = compute_lyapunov(system)
    
    results[system] = {
        "lambdas": [float(l) for l in lambdas],
        "phi_dyn_max": float(lambdas[0])
    }

# ============================================
# VALIDATION
# ============================================

print("\nPhase 8E Results:")
print("=" * 50)

expected = {
    "logistic": 0.49,
    "harmonic": 0.0,
    "lorenz": 0.9
}

for name, data in results.items():
    lam = data["phi_dyn_max"]
    exp = expected[name]
    diff = abs(lam - exp)
    status = "✅" if diff < 0.15 else "⚠️"
    print(f"\n{name}:")
    print(f"  λ_true:   {exp}")
    print(f"  λ_est:    {lam:.4f}")
    print(f"  diff:     {diff:.4f} {status}")

# ============================================
# SAVE
# ============================================

output_path = os.path.join(OUTPUT_DIR, "results.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to: {output_path}")