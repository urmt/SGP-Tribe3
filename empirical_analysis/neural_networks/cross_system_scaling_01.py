#!/usr/bin/env python3
"""
SFH-SGP_CROSS_SYSTEM_SCALING_01
Test α scaling exponent across multiple dynamical systems
"""
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

np.random.seed(42)

sigma = 1e-3
epsilon_list = np.logspace(-4, -1, 15)
n_total = 6000
n_burn = 2000

def recurrence_rate(x, epsilon):
    x = x.reshape(-1, 1)
    D = pairwise_distances(x)
    N = len(x)
    mask = (D < epsilon) & (D > 0)
    return np.sum(mask) / (N * N - N)

def logistic(r, n, x0=0.5):
    x = x0
    xs = []
    for _ in range(n):
        x = r * x * (1 - x)
        xs.append(x)
    return np.array(xs)

def henon_map(a, b, n, x0=0.1, y0=0.1):
    x, y = x0, y0
    xs = []
    for _ in range(n):
        x_new = 1 - a * x**2 + y
        y_new = b * x
        x, y = x_new, y_new
        xs.append(x)
    return np.array(xs)

def kuramoto(K, n, n_osc=200):
    theta = np.random.uniform(0, 2*np.pi, n_osc)
    for _ in range(n):
        d_theta = K * np.sin(theta - theta.reshape(-1, 1))
        d_theta = d_theta.mean(axis=1)
        theta = theta + d_theta
    return theta

def ou_process(theta, n, mu=0, sigma=1.0, dt=0.01):
    x = np.zeros(n)
    x[0] = np.random.randn()
    for t in range(1, n):
        x[t] = x[t-1] + theta * (mu - x[t-1]) * dt + sigma * np.sqrt(dt) * np.random.randn()
    return x

systems = {
    "logistic_periodic": lambda: logistic(3.5, n_total)[n_burn:],
    "logistic_chaotic": lambda: logistic(3.9, n_total)[n_burn:],
    "henon": lambda: henon_map(1.4, 0.3, n_total)[n_burn:],
    "kuramoto_sync": lambda: kuramoto(1.5, n_total),
    "ou": lambda: ou_process(2.0, n_total)
}

results = []

print("Running cross-system scaling test...")

for sys_name, gen_func in systems.items():
    
    x = gen_func()
    x = x + sigma * np.random.randn(len(x))
    
    R = []
    for eps in epsilon_list:
        rr = recurrence_rate(x, eps)
        R.append(rr)
    
    valid = np.array(R) > 0
    if valid.sum() > 2:
        alpha = np.polyfit(np.log(epsilon_list[valid]), np.log(np.array(R)[valid]), 1)[0]
    else:
        alpha = np.nan
    
    results.append({
        "system": sys_name,
        "alpha": alpha,
        "R_at_1e-2": R[list(epsilon_list).index(1e-2)] if 1e-2 in epsilon_list else np.nan
    })

df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("CROSS-SYSTEM SCALING RESULTS")
print("=" * 60)
print(df.to_string(index=False))

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

print("\nα (scaling exponent) by system:")

periodic = ["logistic_periodic"]
chaotic = ["logistic_chaotic", "henon"]
continuous = ["kuramoto_sync"]
stochastic = ["ou"]

for _, row in df.iterrows():
    sys = row["system"]
    if sys in periodic:
        regime = "PERIODIC"
    elif sys in chaotic:
        regime = "CHAOTIC"
    elif sys in continuous:
        regime = "CONTINUOUS"
    else:
        regime = "STOCHASTIC"
    print(f"  {sys:<20}: α = {row['alpha']:.3f} ({regime})")

print("\n" + "=" * 60)
print("GENERALIZATION TEST")
print("=" * 60)

periodic_alphas = df[df["system"].isin(periodic)]["alpha"].mean()
chaotic_alphas = df[df["system"].isin(chaotic)]["alpha"].mean()
continuous_alphas = df[df["system"].isin(continuous)]["alpha"].mean()
stochastic_alphas = df[df["system"].isin(stochastic)]["alpha"].mean()

print(f"\nPeriodic mean α: {periodic_alphas:.3f}")
print(f"Chaotic mean α: {chaotic_alphas:.3f}")
print(f"Continuous mean α: {continuous_alphas:.3f}")
print(f"Stochastic mean α: {stochastic_alphas:.3f}")

if periodic_alphas < chaotic_alphas - 0.2:
    print("\n→ α SEPARATES periodic from chaotic: YES")
    print("→ Generalization: STRONG")
else:
    print("\n→ α SEPARATES periodic from chaotic: NO/MARGINAL")
    print("→ Generalization: WEAK")

df.to_csv("/home/student/sgp-tribe3/empirical_analysis/neural_networks/cross_system_scaling.csv", index=False)
print(f"\nSaved: cross_system_scaling.csv")