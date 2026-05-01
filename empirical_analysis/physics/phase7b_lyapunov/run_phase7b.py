"""
OPENCODE RULES — DO NOT MODIFY

1. DO NOT reduce number of r samples
2. DO NOT reduce signal length
3. DO NOT approximate Lyapunov exponent
4. DO NOT modify epsilon schedule
5. DO NOT subsample time series
6. DO NOT optimize O(n^2) loops

This experiment depends on precision, not speed.
"""

import numpy as np
import json

# =========================
# GLOBAL CONSTANTS
# =========================

N = 1500
TRANSIENT = 500
EPS_STEPS = 20
EPS_MIN = 1e-4
EPS_MAX = 1e-1
L_MIN = 2
VAR_EPS = 1e-8

R_VALUES = np.linspace(3.5, 4.0, 40)

np.random.seed(42)

# =========================
# LOGISTIC MAP
# =========================

def logistic_map(r, x0=0.5):
    x = np.zeros(N)
    x[0] = x0
    for i in range(1, N):
        x[i] = r * x[i-1] * (1 - x[i-1])
    return x[TRANSIENT:]

# =========================
# LYAPUNOV EXPONENT
# =========================

def lyapunov_exponent(r, x):
    lyap = 0.0
    for xi in x:
        lyap += np.log(abs(r * (1 - 2 * xi)) + 1e-12)
    return lyap / len(x)

# =========================
# RECURRENCE
# =========================

def recurrence_matrix(x, eps):
    n = len(x)
    R = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            if abs(x[i] - x[j]) < eps:
                R[i, j] = 1
    return R

def recurrence_rate(R):
    return np.sum(R) / (R.shape[0]**2)

def determinism(R):
    n = R.shape[0]
    diag_lengths = []

    for k in range(-n+1, n):
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

    num = sum(diag_lengths)
    den = np.sum(R)

    return num / den if den > 0 else 0.0

# =========================
# EPSILON
# =========================

def adaptive_eps(x):
    sigma = np.std(x)
    base = np.logspace(np.log10(EPS_MIN), np.log10(EPS_MAX), EPS_STEPS)
    return base * sigma

# =========================
# METRICS
# =========================

from scipy.integrate import trapezoid

def compute_phi(x):

    eps_vals = adaptive_eps(x)

    RR = []
    DET = []

    for eps in eps_vals:
        R = recurrence_matrix(x, eps)
        RR.append(recurrence_rate(R))
        DET.append(determinism(R))

    RR = np.array(RR)
    DET = np.array(DET)

    log_eps = np.log(eps_vals)
    log_rr = np.log(RR + 1e-12)

    alpha = np.polyfit(log_eps, log_rr, 1)[0]

    F = trapezoid(DET, eps_vals)

    var = np.var(DET)
    C = 1.0 / (var + VAR_EPS)

    return alpha, DET.mean(), F, C

# =========================
# MAIN
# =========================

def run():

    results = []

    for r in R_VALUES:

        x = logistic_map(r)

        lyap = lyapunov_exponent(r, x)

        alpha, det, F, C = compute_phi(x)

        results.append({
            "r": float(r),
            "lyapunov": float(lyap),
            "alpha": float(alpha),
            "DET": float(det),
            "F": float(F),
            "C": float(C)
        })

    return results

# =========================
# CORRELATION
# =========================

def compute_correlations(data):

    lyap = np.array([d["lyapunov"] for d in data])
    alpha = np.array([d["alpha"] for d in data])
    F = np.array([d["F"] for d in data])
    DET = np.array([d["DET"] for d in data])

    def corr(x, y):
        return float(np.corrcoef(x, y)[0,1])

    return {
        "alpha_vs_lyapunov": corr(alpha, lyap),
        "F_vs_lyapunov": corr(F, lyap),
        "DET_vs_lyapunov": corr(DET, lyap)
    }

# =========================
# EXECUTE
# =========================

if __name__ == "__main__":

    data = run()

    correlations = compute_correlations(data)

    output = {
        "data": data,
        "correlations": correlations
    }

    path = "empirical_analysis/physics/phase7b_lyapunov/phase7b_results.json"

    with open(path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Saved to: {path}")
    print("Correlations:", correlations)