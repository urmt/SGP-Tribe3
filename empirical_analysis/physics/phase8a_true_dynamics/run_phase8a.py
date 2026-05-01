"""
PHASE 8A — TRUE DYNAMICS OBSERVABLE

STRICT RULES:
- DO NOT reduce trajectory length
- DO NOT approximate derivatives
- DO NOT reuse recurrence code from earlier phases
"""

import numpy as np
import json

N = 2000
TRANSIENT = 500

def logistic(r=3.9):
    x = np.zeros(N)
    x[0] = 0.5
    for i in range(1, N):
        x[i] = r * x[i-1] * (1 - x[i-1])
    return x[TRANSIENT:]

def harmonic():
    t = np.linspace(0, 40, N)
    return np.sin(t)[TRANSIENT:]

def lorenz(dt=0.01):
    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)
    x[0], y[0], z[0] = 1, 1, 1
    for i in range(1, N):
        dx = 10 * (y[i-1] - x[i-1])
        dy = x[i-1] * (28 - z[i-1]) - y[i-1]
        dz = x[i-1] * y[i-1] - (8/3) * z[i-1]
        x[i] = x[i-1] + dx * dt
        y[i] = y[i-1] + dy * dt
        z[i] = z[i-1] + dz * dt
    return np.stack([x, y, z], axis=1)[TRANSIENT:]

def estimate_lyapunov(x):
    diffs = np.abs(np.diff(x))
    diffs = diffs[diffs > 1e-12]
    if len(diffs) == 0:
        return 0.0
    return np.mean(np.log(diffs))

def entropy_rate(x):
    hist, _ = np.histogram(x, bins=30, density=True)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.0
    return -np.sum(hist * np.log(hist + 1e-12))

def correlation_dimension(data, r_min=0.01, r_max=1.0, n_bins=20):
    radii = np.logspace(np.log10(r_min), np.log10(r_max), n_bins)
    counts = []
    for r in radii:
        count = 0
        for i in range(len(data)):
            for j in range(i+1, len(data)):
                if np.linalg.norm(data[i] - data[j]) < r:
                    count += 1
        counts.append(count)
    counts = np.array(counts, dtype=float)
    radii = np.array(radii)
    valid = counts > 0
    if np.sum(valid) < 2:
        return 0.0
    slope = np.polyfit(np.log(radii[valid]), np.log(counts[valid]), 1)[0]
    return slope

results = []

# Test harmonic
x_harm = harmonic()
results.append({
    "system": "harmonic",
    "lyapunov": float(estimate_lyapunov(x_harm)),
    "entropy": float(entropy_rate(x_harm)),
    "D2": float(correlation_dimension(x_harm.reshape(-1, 1)))
})

# Test logistic
x_log = logistic()
results.append({
    "system": "logistic",
    "lyapunov": float(estimate_lyapunov(x_log)),
    "entropy": float(entropy_rate(x_log)),
    "D2": float(correlation_dimension(x_log.reshape(-1, 1)))
})

# Test lorenz
x_lor = lorenz()
results.append({
    "system": "lorenz",
    "lyapunov": float(estimate_lyapunov(x_lor[:, 0])),
    "entropy": float(entropy_rate(x_lor[:, 0])),
    "D2": float(correlation_dimension(x_lor))
})

path = "empirical_analysis/physics/phase8a_true_dynamics/results.json"

with open(path, "w") as f:
    json.dump(results, f, indent=4)

print(f"Saved: {path}")
for r in results:
    print(r)