#!/usr/bin/env python3
"""
SFH-SGP_DUAL_COLLAPSE_TEST_01
Test whether dimensional collapse occurs in multiple systems
"""
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

EMBED_DIM = 10
DELAY = 2
K_VALUES = [2, 4, 8, 16]
OUT_DIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/"

np.random.seed(42)

def compute_knn_dim(data, k):
    try:
        n, d = data.shape
        if n <= k + 1 or d < 2:
            return np.nan
        data_clean = data.copy()
        if np.std(data_clean) < 1e-10:
            return np.nan
        nn = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree')
        nn.fit(data_clean)
        dists, _ = nn.kneighbors(data_clean)
        return np.mean(np.log(dists[:, -1] + 1e-10))
    except:
        return np.nan

def time_delay_embed(x, dim, delay):
    N = len(x) - (dim - 1) * delay
    embedded = np.zeros((N, dim))
    for i in range(dim):
        embedded[:, i] = x[i*delay : i*delay + N]
    return embedded

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("SYSTEM 1: KURAMOTO MODEL")
print("=" * 70)

N_OSC = 100
T_STEPS = 5000
DT = 0.05
TRANSIENT = 1000
K_VALS = np.arange(0, 3.2, 0.2)

kur_results = []

for K in np.round(K_VALS, 1):
    np.random.seed(42 + int(K*10))
    
    omega = np.random.randn(N_OSC)
    theta = np.random.uniform(0, 2*np.pi, N_OSC)
    
    order_param = np.zeros(T_STEPS + TRANSIENT)
    
    for t in range(T_STEPS + TRANSIENT):
        coupling = (K / N_OSC) * np.sum(np.sin(theta[:, np.newaxis] - theta), axis=1)
        dtheta = omega + coupling
        theta = theta + DT * dtheta
        theta = theta % (2 * np.pi)
        
        if t >= TRANSIENT:
            r = np.abs(np.mean(np.exp(1j * theta)))
            order_param[t - TRANSIENT] = r
    
    signal = order_param[:T_STEPS]
    signal = (signal - signal.mean()) / (signal.std() + 1e-10)
    
    embedded = time_delay_embed(signal, EMBED_DIM, DELAY)
    
    dk = {}
    for k in K_VALUES:
        dk[k] = compute_knn_dim(embedded, k)
    
    kur_results.append({
        'param': K,
        'D2': dk[2],
        'D4': dk[4],
        'D8': dk[8],
        'D16': dk[16]
    })

df_kur = pd.DataFrame(kur_results).dropna()

dk_cols = ['D2', 'D4', 'D8', 'D16']
df_kur['dk_std'] = df_kur[dk_cols].std(axis=1)

print(f"\nKuramoto: {len(df_kur)} points")
print(f"{'K':<6} {'D2':<10} {'D4':<10} {'D8':<10} {'D16':<10} {'std':<10}")
print("-" * 60)
for _, row in df_kur.iterrows():
    print(f"{row['param']:<6.1f} {row['D2']:<10.4f} {row['D4']:<10.4f} {row['D8']:<10.4f} {row['D16']:<10.4f} {row['dk_std']:<10.4f}")

print("\n" + "=" * 70)
print("SYSTEM 2: ORNSTEIN-UHLENBECK")
print("=" * 70)

N_SAMPLES = 5000
TRANSIENT = 1000
DT = 0.01
THETA_VALS = np.arange(0, 2.2, 0.2)

ou_results = []

for theta in np.round(THETA_VALS, 1):
    x = 0.0
    sigma = 1.0
    
    samples = np.zeros(N_SAMPLES + TRANSIENT)
    for i in range(N_SAMPLES + TRANSIENT):
        dW = np.random.randn() * np.sqrt(DT)
        dx = -theta * x * DT + sigma * dW
        x = x + dx
        samples[i] = x
    
    signal = samples[TRANSIENT:]
    signal = (signal - signal.mean()) / (signal.std() + 1e-10)
    
    embedded = time_delay_embed(signal, EMBED_DIM, DELAY)
    
    dk = {}
    for k in K_VALUES:
        dk[k] = compute_knn_dim(embedded, k)
    
    ou_results.append({
        'param': theta,
        'D2': dk[2],
        'D4': dk[4],
        'D8': dk[8],
        'D16': dk[16]
    })

df_ou = pd.DataFrame(ou_results).dropna()

df_ou['dk_std'] = df_ou[dk_cols].std(axis=1)

print(f"\nOU: {len(df_ou)} points")
print(f"{'theta':<8} {'D2':<10} {'D4':<10} {'D8':<10} {'D16':<10} {'std':<10}")
print("-" * 65)
for _, row in df_ou.iterrows():
    print(f"{row['param']:<8.1f} {row['D2']:<10.4f} {row['D4']:<10.4f} {row['D8']:<10.4f} {row['D16']:<10.4f} {row['dk_std']:<10.4f}")

print("\n" + "=" * 70)
print("STEP 3-4: VARIANCE ANALYSIS")
print("=" * 70)

print("\n--- KURAMOTO ---")
kur_flat = df_kur[df_kur['dk_std'] < 0.1]
print(f"Points with std < 0.1: {len(kur_flat)} / {len(df_kur)}")

print("\n--- OU ---")
ou_flat = df_ou[df_ou['dk_std'] < 0.1]
print(f"Points with std < 0.1: {len(ou_flat)} / {len(df_ou)}")

print("\n" + "=" * 70)
print("COMPARISON: Logistic vs Kuramoto vs OU")
print("=" * 70)

print(f"\n{'System':<12} {'Flat regions':<15} {'Mean std':<12} {'Min D(k)':<12}")
print("-" * 55)

log_std = 0.0
print(f"{'Logistic':<12} {'~r<3.55':<15} {'0.0':<12} {'-23.03':<12}")
print(f"{'Kuramoto':<12} {len(kur_flat):<15} {df_kur['dk_std'].mean():<12.4f} {df_kur['D2'].min():<12.4f}")
print(f"{'OU':<12} {len(ou_flat):<15} {df_ou['dk_std'].mean():<12.4f} {df_ou['D2'].min():<12.4f}")

print("\n" + "=" * 70)
print("FINAL QUESTIONS")
print("=" * 70)

any_flat_kur = len(kur_flat) > 0
any_flat_ou = len(ou_flat) > 0

print(f"\n1. Any approach zero?")
print(f"   Kuramoto: {'YES' if any_flat_kur else 'NO'}")
print(f"   OU: {'YES' if any_flat_ou else 'NO'}")

specific_kur = any_flat_kur and len(kur_flat) < len(df_kur) * 0.5
specific_ou = any_flat_ou and len(ou_flat) < len(df_ou) * 0.5

print(f"\n2. Only in specific regions?")
print(f"   Kuramoto: {'YES' if specific_kur else 'NO'}")
print(f"   OU: {'YES' if specific_ou else 'NO'}")

unique_log = (log_std == 0) and not any_flat_kur and not any_flat_ou

print(f"\n3. Unique to logistic? {'YES' if unique_log else 'NO'}")

df_kur.to_csv(os.path.join(OUT_DIR, "collapse_kuramoto.csv"), index=False)
df_ou.to_csv(os.path.join(OUT_DIR, "collapse_ou.csv"), index=False)
print(f"\nSaved: collapse_kuramoto.csv, collapse_ou.csv")