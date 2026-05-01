import numpy as np
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

T = 3000
T_transient = 1000
eps = 0.05

def logistic_step(x, r):
    return r * x * (1 - x)

def harmonic_step(x, v, dt=0.01):
    k = 1.0
    x_new = x + v * dt
    v_new = v - k * x * dt
    return x_new, v_new

def lorenz_step(x, y, z, dt=0.01):
    sigma = 10
    rho = 28
    beta = 8/3
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return x + dx*dt, y + dy*dt, z + dz*dt

def lambda_logistic(r, traj):
    vals = np.log(np.abs(r * (1 - 2*traj)) + 1e-12)
    return np.mean(vals)

def lambda_lorenz():
    return 0.905

def lambda_harmonic():
    return 0.0

def compute_FC(signal):
    signal = signal.reshape(-1, 1)
    N = len(signal)
    D = np.abs(signal - signal.T)
    R = D < eps
    DET = np.mean(R)
    F = DET
    var = np.var(np.mean(R, axis=1))
    C = 1.0 / (var + 1e-8)
    return F, C

logistic_data = []
lorenz_data = []
harmonic_data = []

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
    logistic_data.append([F, C, lam])

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
    lorenz_data.append([F, C, lam])

for seed in range(10):
    x, v = 1.0 + seed*0.01, 0.0
    traj = []
    for t in range(T):
        x, v = harmonic_step(x, v)
        if t > T_transient:
            traj.append(x)
    traj = np.array(traj)
    lam = lambda_harmonic()
    F, C = compute_FC(traj)
    harmonic_data.append([F, C, lam])

logistic_data = np.array(logistic_data)
X_train = logistic_data[:, :2]
y_train = logistic_data[:, 2]

model = LinearRegression().fit(X_train, y_train)

def evaluate(data, name):
    data = np.array(data)
    X = data[:, :2]
    y_true = data[:, 2]
    y_pred = model.predict(X)
    r, _ = pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"r": float(r), "r2": float(r2)}

results = {
    "train_logistic": evaluate(logistic_data, "logistic"),
    "test_lorenz": evaluate(lorenz_data, "lorenz"),
    "test_harmonic": evaluate(harmonic_data, "harmonic")
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== PHASE 9D RESULTS ===")
for k, v in results.items():
    print(k, "-> r:", round(v["r"], 3), "R²:", round(v["r2"], 3))