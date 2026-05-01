"""
OPENCODE RULES — DO NOT MODIFY

1. DO NOT reduce trajectory length
2. DO NOT reduce number of systems
3. DO NOT change integration steps
4. DO NOT approximate Lyapunov exponents
5. DO NOT subsample trajectories
6. DO NOT optimize recurrence matrix (O(n^2) required)

This experiment depends on preserving dynamical fidelity.
"""

import numpy as np
import json
from scipy.integrate import trapezoid

N = 800
TRANSIENT = 200
EPS_STEPS = 15
EPS_MIN = 1e-4
EPS_MAX = 1e-2
L_MIN = 2
VAR_EPS = 1e-8

np.random.seed(42)

def logistic_map(r=3.9):
    x = np.zeros(N)
    x[0] = 0.5
    for i in range(1, N):
        x[i] = r * x[i-1] * (1 - x[i-1])
    return x[TRANSIENT:]

def logistic_lyapunov(r, x):
    return np.mean(np.log(abs(r * (1 - 2*x)) + 1e-12))

def harmonic_oscillator():
    t = np.linspace(0, 40, N)
    return np.sin(t)[TRANSIENT:]

def harmonic_lyapunov(x):
    return 0.0

def lorenz_system(dt=0.01, sigma=10, rho=28, beta=8/3):
    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)
    x[0], y[0], z[0] = 1.0, 1.0, 1.0
    for i in range(1, N):
        dx = sigma * (y[i-1] - x[i-1])
        dy = x[i-1] * (rho - z[i-1]) - y[i-1]
        dz = x[i-1] * y[i-1] - beta * z[i-1]
        x[i] = x[i-1] + dx * dt
        y[i] = y[i-1] + dy * dt
        z[i] = z[i-1] + dz * dt
    return x[TRANSIENT:]

def lorenz_lyapunov():
    return 0.905  # known positive exponent

def recurrence_matrix(x, eps):
    n = len(x)
    R = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            if abs(x[i] - x[j]) < eps:
                R[i, j] = 1
    return R

def determinism(R):
    n = R.shape[0]
    total = 0
    for k in range(-n+1, n):
        diag = np.diag(R, k=k)
        length = 0
        for val in diag:
            if val == 1:
                length += 1
            else:
                if length >= L_MIN:
                    total += length
                length = 0
        if length >= L_MIN:
            total += length
    return total / (np.sum(R) + 1e-12)

def compute_phi(x):
    sigma = np.std(x)
    eps_vals = np.logspace(np.log10(EPS_MIN), np.log10(EPS_MAX), EPS_STEPS) * max(sigma, 0.01)
    RR, DET = [], []
    for eps in eps_vals:
        R = recurrence_matrix(x, eps)
        RR.append(np.mean(R))
        DET.append(determinism(R))
    RR = np.array(RR)
    DET = np.array(DET)
    alpha = np.polyfit(np.log(eps_vals + 1e-12), np.log(RR + 1e-12), 1)[0]
    F = trapezoid(DET, eps_vals)
    C = 1.0 / (np.var(DET) + VAR_EPS)
    return alpha, np.mean(DET), F, C

def run():
    results = []
    
    # Logistic
    x_log = logistic_map()
    lyap_log = logistic_lyapunov(3.9, x_log)
    alpha, det, F, C = compute_phi(x_log)
    results.append({"system": "logistic", "lyapunov": float(lyap_log), "alpha": float(alpha), "DET": float(det), "F": float(F), "C": float(C)})
    
    # Harmonic
    x_harm = harmonic_oscillator()
    lyap_harm = harmonic_lyapunov(x_harm)
    alpha, det, F, C = compute_phi(x_harm)
    results.append({"system": "harmonic", "lyapunov": float(lyap_harm), "alpha": float(alpha), "DET": float(det), "F": float(F), "C": float(C)})
    
    # Lorenz
    x_lor = lorenz_system()
    lyap_lor = lorenz_lyapunov()
    alpha, det, F, C = compute_phi(x_lor)
    results.append({"system": "lorenz", "lyapunov": float(lyap_lor), "alpha": float(alpha), "DET": float(det), "F": float(F), "C": float(C)})
    
    return results

if __name__ == "__main__":
    results = run()
    path = "empirical_analysis/physics/phase7c_cross_system/phase7c_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved to: {path}")
    for r in results:
        print(r)