"""
OPENCODE EXECUTION RULES — DO NOT MODIFY

1. DO NOT reduce signal length
2. DO NOT reduce epsilon resolution
3. DO NOT approximate recurrence computation
4. DO NOT subsample time series
5. DO NOT change any constants
6. If runtime is slow, WAIT — do not optimize

This script is intentionally O(n^2). That is REQUIRED.

Violating these rules invalidates the experiment.
"""

import numpy as np
import json
from scipy.integrate import odeint
from scipy.integrate import trapezoid

# =========================
# FIXED GLOBAL PARAMETERS
# =========================

N = 1000                  # FIXED signal length
EPS_STEPS = 20            # MUST NOT CHANGE
EPS_MIN = 1e-4
EPS_MAX = 1e-1
L_MIN = 2
VAR_EPS = 1e-8            # prevents C explosion

np.random.seed(42)

# =========================
# SYSTEM DEFINITIONS
# =========================

def logistic_map(r=3.9, x0=0.2):
    x = np.zeros(N)
    x[0] = x0
    for i in range(1, N):
        x[i] = r * x[i-1] * (1 - x[i-1])
    return x

def harmonic_oscillator():
    t = np.linspace(0, 20*np.pi, N)
    return np.sin(t)

def lorenz_system():
    def lorenz(state, t):
        x, y, z = state
        sigma, rho, beta = 10, 28, 8/3
        return [
            sigma*(y - x),
            x*(rho - z) - y,
            x*y - beta*z
        ]
    t = np.linspace(0, 20, N)
    states = odeint(lorenz, [1,1,1], t)
    return states[:,0]

# =========================
# OPERATORS
# =========================

def identity(x):
    return x

def tanh_1(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def softplus(x):
    return np.log1p(np.exp(x))

OPERATORS = {
    "identity": identity,
    "tanh_1": tanh_1,
    "relu": relu,
    "softplus": softplus
}

# =========================
# RECURRENCE ANALYSIS
# =========================

def recurrence_matrix(x, eps):
    N = len(x)
    R = np.zeros((N, N), dtype=np.uint8)
    for i in range(N):
        for j in range(N):
            if abs(x[i] - x[j]) < eps:
                R[i, j] = 1
    return R

def recurrence_rate(R):
    return np.sum(R) / (R.shape[0]**2)

def determinism(R):
    N = R.shape[0]
    diag_lengths = []

    for k in range(-N+1, N):
        diag = np.diag(R, k=k)
        length = 0
        for val in diag:
            if val == 1:
                length += 1
            else:
                if length >= L_MIN:
                    diag_lengths.append(length)
                length = 0
        if length >= L_MIN:
            diag_lengths.append(length)

    if len(diag_lengths) == 0:
        return 0.0

    num = sum(l for l in diag_lengths)
    den = np.sum(R)

    return num / den if den > 0 else 0.0

# =========================
# EPSILON (CRITICAL FIX)
# =========================

def adaptive_epsilons(x):
    sigma = np.std(x)
    base = np.logspace(np.log10(EPS_MIN), np.log10(EPS_MAX), EPS_STEPS)
    return base * sigma

# =========================
# METRICS
# =========================

def compute_metrics(x):

    eps_vals = adaptive_epsilons(x)

    RR = []
    DET = []

    for eps in eps_vals:
        R = recurrence_matrix(x, eps)
        RR.append(recurrence_rate(R))
        DET.append(determinism(R))

    RR = np.array(RR)
    DET = np.array(DET)

    # α: slope of log-log RR
    log_eps = np.log(eps_vals)
    log_rr = np.log(RR + 1e-12)
    alpha = np.polyfit(log_eps, log_rr, 1)[0]

    # Fertility (F)
    F = trapezoid(DET, eps_vals)

    # Coherence (C) with stabilization
    variance = np.var(DET)
    C = 1.0 / (variance + VAR_EPS)

    return alpha, DET.mean(), F, C, DET.tolist()

# =========================
# MAIN EXECUTION
# =========================

def run():

    systems = {
        "logistic": logistic_map(),
        "harmonic": harmonic_oscillator(),
        "lorenz": lorenz_system()
    }

    results = {}

    for sys_name, signal in systems.items():
        results[sys_name] = {}

        for op_name, op in OPERATORS.items():

            x = op(signal)

            alpha, det, F, C, det_curve = compute_metrics(x)

            results[sys_name][op_name] = {
                "alpha": float(alpha),
                "DET": float(det),
                "F": float(F),
                "C": float(C),
                "DET_curve": det_curve
            }

    return results

# =========================
# SAVE OUTPUT
# =========================

if __name__ == "__main__":

    results = run()

    output_path = "empirical_analysis/physics/phase7a_dynamical_systems/phase7a_corrected_results.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to: {output_path}")