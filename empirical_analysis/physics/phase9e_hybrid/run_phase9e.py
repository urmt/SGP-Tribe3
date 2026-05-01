import numpy as np
import json
from sklearn.linear_model import LinearRegression
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

def compute_features(signal):
    signal = np.array(signal)
    N = len(signal)

    X = signal.reshape(-1, 1)
    D = np.abs(X - X.T)
    R = D < eps

    F = np.mean(R)

    var = np.var(np.mean(R, axis=1))
    C = 1.0 / (var + 1e-8)

    v = np.diff(signal)
    v_mean = np.mean(np.abs(v))

    dv = np.diff(signal)
    dv2 = np.diff(signal, n=2)

    local_div = np.mean(np.abs(dv2) / (np.abs(dv[:-1]) + 1e-8))

    return F, C, v_mean, local_div

logistic_data = []
lorenz_data = []
harmonic_data = []

r_values = np.linspace(3.5, 4.0, 30)

for r in r_values:
    x = 0.5
    traj = []
    for t in range(T):
        x = logistic_step(x, r)
        if t > T_transient:
            traj.append(x)
    traj = np.array(traj)
    lam = lambda_logistic(r, traj)
    F, C, v, div = compute_features(traj)
    logistic_data.append([F, C, v, div, lam])

for seed in range(10):
    x, y, z = 1.0 + seed*0.01, 1.0, 1.0
    traj = []
    for t in range(T):
        x, y, z = lorenz_step(x, y, z)
        if t > T_transient:
            traj.append(x)
    traj = np.array(traj)
    lam = lambda_lorenz()
    F, C, v, div = compute_features(traj)
    lorenz_data.append([F, C, v, div, lam])

for seed in range(10):
    x, v0 = 1.0 + seed*0.01, 0.0
    traj = []
    for t in range(T):
        x, v0 = harmonic_step(x, v0)
        if t > T_transient:
            traj.append(x)
    traj = np.array(traj)
    lam = lambda_harmonic()
    F, C, v, div = compute_features(traj)
    harmonic_data.append([F, C, v, div, lam])

logistic_data = np.array(logistic_data)
X_train = logistic_data[:, :4]
y_train = logistic_data[:, 4]

model = LinearRegression().fit(X_train, y_train)

def evaluate(data):
    data = np.array(data)
    X = data[:, :4]
    y_true = data[:, 4]
    y_pred = model.predict(X)
    r, _ = pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return r, r2

results = {
    "train_logistic": evaluate(logistic_data),
    "test_lorenz": evaluate(lorenz_data),
    "test_harmonic": evaluate(harmonic_data)
}

with open("results.json", "w") as f:
    json.dump({k: {"r": float(v[0]), "r2": float(v[1])} for k, v in results.items()}, f, indent=2)

print("\n=== PHASE 9E RESULTS ===")
for k, v in results.items():
    print(k, "-> r:", round(v[0], 3), "R²:", round(v[1], 3))