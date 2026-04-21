#!/usr/bin/env python3
"""
SFH-SGP_ALPHA_DET_COMBINED_01
Combined observable: α (scaling) + DET (determinism)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

epsilon = 1e-2
l_min = 2

def logistic(r, x0, n):
    x = x0
    xs = []
    for _ in range(n):
        x = r * x * (1 - x)
        xs.append(x)
    return np.array(xs)

def henon(a, b, x0, y0, n):
    x, y = x0, y0
    xs = []
    for _ in range(n):
        x_new = 1 - a * x**2 + y
        y = b * x
        x = x_new
        xs.append(x)
    return np.array(xs)

def ou_process(theta, n):
    x = 0
    xs = []
    for _ in range(n):
        x = x + theta * (-x) + np.random.randn() * 0.1
        xs.append(x)
    return np.array(xs)

def recurrence_matrix(x, epsilon):
    x = x.reshape(-1, 1)
    D = pairwise_distances(x)
    R = (D < epsilon).astype(int)
    np.fill_diagonal(R, 0)
    return R

def compute_det(R, l_min=2):
    N = R.shape[0]
    diag_counts = 0
    total_rec = np.sum(R)
    
    for k in range(-N+1, N):
        diag = np.diagonal(R, offset=k)
        
        length = 0
        for val in diag:
            if val == 1:
                length += 1
            else:
                if length >= l_min:
                    diag_counts += length
                length = 0
        
        if length >= l_min:
            diag_counts += length
    
    if total_rec == 0:
        return 0.0
    
    return diag_counts / total_rec

def recurrence_rate(R):
    N = R.shape[0]
    return np.sum(R) / (N * N - N)

def compute_alpha(x, eps_list):
    R_vals = []
    
    for eps in eps_list:
        R = recurrence_matrix(x, eps)
        R_vals.append(recurrence_rate(R))
    
    log_eps = np.log(eps_list)
    log_R = np.log(R_vals)
    
    valid = np.isfinite(log_R)
    if valid.sum() > 2:
        alpha = np.polyfit(log_eps[valid], log_R[valid], 1)[0]
    else:
        alpha = np.nan
    
    return alpha

n_total = 6000
n_burn = 2000
sigma = 1e-3

eps_list = np.logspace(-4, -1, 12)

results = []

print("Running α + DET combined test...")

systems = [
    ("logistic_periodic", lambda: logistic(3.5, np.random.rand(), n_total)),
    ("logistic_chaotic", lambda: logistic(3.9, np.random.rand(), n_total)),
    ("henon", lambda: henon(1.4, 0.3, 0.1, 0.1, n_total)),
    ("ou", lambda: ou_process(0.5, n_total))
]

for name, generator in systems:
    
    x = generator()[n_burn:]
    x = x + sigma * np.random.randn(len(x))
    
    alpha = compute_alpha(x, eps_list)
    
    R = recurrence_matrix(x, epsilon)
    det = compute_det(R, l_min=2)
    
    results.append({
        "system": name,
        "alpha": alpha,
        "DET": det
    })

df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("ALPHA + DET RESULTS")
print("=" * 60)
print(df.to_string(index=False))

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

print("\nExpected:")
print("        ↑ DET")
print("        |")
print("Periodic | Continuous")
print("        |")
print("--------+--------→ α")
print("        |")
print("Chaotic | Stochastic")

print("\nObserved:")
for _, row in df.iterrows():
    print(f"  {row['system']:<20}: α={row['alpha']:.3f}, DET={row['DET']:.3f}")

print("\n" + "=" * 60)
print("SEPARATION ANALYSIS")
print("=" * 60)

periodic = df[df["system"] == "logistic_periodic"]
chaotic = df[df["system"].isin(["logistic_chaotic", "henon"])]
stochastic = df[df["system"] == "ou"]

print(f"\nPeriodic: α={periodic['alpha'].values[0]:.3f}, DET={periodic['DET'].values[0]:.3f}")
print(f"Chaotic: α={chaotic['alpha'].mean():.3f}, DET={chaotic['DET'].mean():.3f}")
print(f"Stochastic: α={stochastic['alpha'].values[0]:.3f}, DET={stochastic['DET'].values[0]:.3f}")

df.to_csv("/home/student/sgp-tribe3/empirical_analysis/neural_networks/alpha_det_results.csv", index=False)
print(f"\nSaved: alpha_det_results.csv")