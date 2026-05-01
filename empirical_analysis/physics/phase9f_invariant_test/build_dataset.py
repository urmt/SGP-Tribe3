import numpy as np
import json
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

T = 3000
T_transient = 1000
eps = 0.05
dt = 0.01

def logistic_step(x, r):
    return r * x * (1 - x)

def harmonic_step(x, v):
    k = 1.0
    x_new = x + v * dt
    v_new = v - k * x * dt
    return x_new, v_new

def lorenz_step(x, y, z):
    sigma = 10
    rho = 28
    beta = 8/3
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return x + dx*dt, y + dy*dt, z + dz*dt

def lambda_logistic(r, traj):
    return np.mean(np.log(np.abs(r * (1 - 2*traj)) + 1e-12))

def lambda_lorenz():
    return 0.905

def lambda_harmonic():
    return 0.0

def compute_FC(signal):
    X = signal.reshape(-1, 1)
    D = np.abs(X - X.T)
    R = D < eps
    F = np.mean(R)
    var = np.var(np.mean(R, axis=1))
    C = 1.0 / (var + 1e-8)
    return F, C

def compute_velocity(signal):
    v = np.diff(signal)
    return np.mean(np.abs(v))

def compute_divergence(signal):
    dv = np.diff(signal)
    dv2 = np.diff(signal, n=2)
    return np.mean(np.abs(dv2) / (np.abs(dv[:-1]) + 1e-8))

print("Building multi-system dataset...")

all_data = []

r_values = np.linspace(3.5, 4.0, 25)
for r in r_values:
    x = 0.5
    traj = []
    for t in range(T):
        x = logistic_step(x, r)
        if t > T_transient:
            traj.append(x)
    traj = np.array(traj)
    lam = lambda_logistic(r, traj)
    F, C = compute_FC(traj)
    v = compute_velocity(traj)
    div = compute_divergence(traj)
    all_data.append({
        "system": "logistic",
        "r": float(r),
        "operator": "identity",
        "F": float(F),
        "C": float(C),
        "velocity": float(v),
        "divergence": float(div),
        "lambda": float(lam)
    })

for seed in range(10):
    x, y, z = 1.0 + seed*0.01, 1.0, 1.0
    traj = []
    for t in range(T):
        x, y, z = lorenz_step(x, y, z)
        if t > T_transient:
            traj.append(x)
    traj = np.array(traj)
    lam = lambda_lorenz()
    F, C = compute_FC(traj)
    v = compute_velocity(traj)
    div = compute_divergence(traj)
    all_data.append({
        "system": "lorenz",
        "r": float(seed),
        "operator": "identity",
        "F": float(F),
        "C": float(C),
        "velocity": float(v),
        "divergence": float(div),
        "lambda": float(lam)
    })

for seed in range(10):
    x, v0 = 1.0 + seed*0.01, 0.0
    traj = []
    for t in range(T):
        x, v0 = harmonic_step(x, v0)
        if t > T_transient:
            traj.append(x)
    traj = np.array(traj)
    lam = lambda_harmonic()
    F, C = compute_FC(traj)
    v = compute_velocity(traj)
    div = compute_divergence(traj)
    all_data.append({
        "system": "harmonic",
        "r": float(seed),
        "operator": "identity",
        "F": float(F),
        "C": float(C),
        "velocity": float(v),
        "divergence": float(div),
        "lambda": float(lam)
    })

with open("input_multisystem.json", "w") as f:
    json.dump(all_data, f, indent=2)

print(f"Dataset built: {len(all_data)} samples")