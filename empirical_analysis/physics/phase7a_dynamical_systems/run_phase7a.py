import os
import json
import numpy as np

SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = "empirical_analysis/physics/phase7a_dynamical_systems/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPSILONS = np.logspace(-3, -1, 10)
L_MIN = 2

def logistic_map(r=3.9, x0=0.5, n=500, discard=100):
    x = np.zeros(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i-1] * (1 - x[i-1])
    return x[discard:]

def harmonic_oscillator(n=500, dt=0.05):
    t = np.arange(0, n*dt, dt)
    x = np.sin(2 * np.pi * t)
    return x

def lorenz_system(n=1000, dt=0.01, sigma=10, rho=28, beta=8/3):
    xs = np.zeros(n)
    ys = np.zeros(n)
    zs = np.zeros(n)
    xs[0], ys[0], zs[0] = 0., 1., 1.05
    for i in range(n - 1):
        x, y, z = xs[i], ys[i], zs[i]
        xs[i+1] = x + sigma*(y - x)*dt
        ys[i+1] = y + (x*(rho - z) - y)*dt
        zs[i+1] = z + (x*y - beta*z)*dt
    return xs[200:]

def identity(x): return x
def tanh_k(x, k=1.0): return np.tanh(k * x)
def relu(x): return np.maximum(0, x)
def softplus(x): return np.log1p(np.exp(x))
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def scale(x, k=2.0): return k * x

OPERATORS = {
    "identity": identity,
    "tanh_1": lambda x: tanh_k(x, 1.0),
    "tanh_2": lambda x: tanh_k(x, 2.0),
    "relu": relu,
    "softplus": softplus,
    "sigmoid": sigmoid,
    "scale_2": lambda x: scale(x, 2.0)
}

def recurrence_matrix(x, eps):
    dist = np.abs(x[:, None] - x[None, :])
    return (dist < eps).astype(int)

def recurrence_rate(R):
    return np.mean(R)

def compute_alpha(x):
    rates = []
    for eps in EPSILONS:
        R = recurrence_matrix(x, eps)
        rates.append(recurrence_rate(R))
    rates = np.array(rates)
    if np.any(rates <= 0):
        return 0.0
    log_eps = np.log(EPSILONS)
    log_rates = np.log(rates)
    slope = np.polyfit(log_eps, log_rates, 1)[0]
    return slope

def compute_det(R):
    N = R.shape[0]
    det_points = 0
    total_points = np.sum(R)
    for k in range(-N+1, N):
        diag = np.diag(R, k)
        length = 0
        for val in diag:
            if val == 1:
                length += 1
            else:
                if length >= L_MIN:
                    det_points += length
                length = 0
        if length >= L_MIN:
            det_points += length
    if total_points == 0:
        return 0.0
    return det_points / total_points

def compute_det_curve(x):
    det_vals = []
    lambdas = np.linspace(0, 2, 10)
    for lam in lambdas:
        x_trans = np.tanh(lam * x)
        R = recurrence_matrix(x_trans, 0.05)
        det_vals.append(compute_det(R))
    return np.array(det_vals), lambdas

def compute_F(det_vals, lambdas):
    return np.trapezoid(det_vals, lambdas)

def compute_C(det_vals):
    var = np.var(det_vals)
    if var == 0:
        return 0.0
    return 1.0 / var

SYSTEMS = {
    "logistic": logistic_map(),
    "harmonic": harmonic_oscillator(),
    "lorenz": lorenz_system()
}

results = {}

for sys_name, signal in SYSTEMS.items():
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        raise ValueError(f"Invalid values in system: {sys_name}")
    sys_results = {}
    for op_name, op_func in OPERATORS.items():
        try:
            x_op = op_func(signal)
        except Exception as e:
            raise RuntimeError(f"Operator failed: {op_name} on {sys_name}") from e
        if np.any(np.isnan(x_op)) or np.any(np.isinf(x_op)):
            raise ValueError(f"Invalid values after operator {op_name} on {sys_name}")
        alpha = compute_alpha(x_op)
        R = recurrence_matrix(x_op, 0.05)
        det = compute_det(R)
        det_curve, lambdas = compute_det_curve(x_op)
        F = compute_F(det_curve, lambdas)
        C = compute_C(det_curve)
        sys_results[op_name] = {
            "alpha": float(alpha),
            "DET": float(det),
            "F": float(F),
            "C": float(C)
        }
    results[sys_name] = sys_results

output_path = os.path.join(OUTPUT_DIR, "phase7a_results.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print("Phase 7A complete.")
print(f"Results saved to: {output_path}")