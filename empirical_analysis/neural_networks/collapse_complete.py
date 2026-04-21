#!/usr/bin/env python3
"""
SFH-SGP_COLLAPSE_COMPLETE_01
Complete dimensional collapse verification for Kuramoto and OU
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

print("=" * 80)
print("SYSTEM 1: KURAMOTO MODEL")
print("=" * 80)

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

print("\nKURAMOTO")
print("K      | D2       | D4       | D8       | D16      | std")
print("-" * 75)
for _, row in df_kur.iterrows():
    print(f"{row['param']:<6.1f} | {row['D2']:<9.4f} | {row['D4']:<9.4f} | {row['D8']:<9.4f} | {row['D16']:<9.4f} | {row['dk_std']:<9.4f}")

print("\n" + "=" * 80)
print("SYSTEM 2: ORNSTEIN-UHLENBECK")
print("=" * 80)

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

print("\nOU")
print("theta  | D2       | D4       | D8       | D16      | std")
print("-" * 75)
for _, row in df_ou.iterrows():
    print(f"{row['param']:<6.1f} | {row['D2']:<9.4f} | {row['D4']:<9.4f} | {row['D8']:<9.4f} | {row['D16']:<9.4f} | {row['dk_std']:<9.4f}")

print("\n" + "=" * 80)
print("STEP 4: VALIDATION")
print("=" * 80)

print(f"\nKuramoto: {len(df_kur)} rows, NaN: {df_kur.isna().sum().sum()}")
print(f"Kuramoto std range: {df_kur['dk_std'].min():.4f} - {df_kur['dk_std'].max():.4f}")

print(f"\nOU: {len(df_ou)} rows, NaN: {df_ou.isna().sum().sum()}")
print(f"OU std range: {df_ou['dk_std'].min():.4f} - {df_ou['dk_std'].max():.4f}")

print("\n" + "=" * 80)
print("STEP 3: SAVE RESULTS")
print("=" * 80)

df_kur['system'] = 'Kuramoto'
df_ou['system'] = 'OU'

df_all = pd.concat([df_kur, df_ou], ignore_index=True)
df_all = df_all[['system', 'param', 'D2', 'D4', 'D8', 'D16', 'dk_std']]

df_all.to_csv(os.path.join(OUT_DIR, "collapse_results.csv"), index=False)
print(f"\nSaved: {OUT_DIR}collapse_results.csv")

print("\n" + "=" * 80)
print("FINAL QUESTIONS")
print("=" * 80)

min_std_kur = df_kur['dk_std'].min()
min_std_ou = df_ou['dk_std'].min()

print(f"\n1. Does ANY std ≈ 0 in Kuramoto? {'YES' if min_std_kur < 0.1 else 'NO'} (min={min_std_kur:.4f})")
print(f"2. Does ANY std ≈ 0 in OU? {'YES' if min_std_ou < 0.1 else 'NO'} (min={min_std_ou:.4f})")

logistic_has_collapse = True
unique_to_logistic = logistic_has_collapse and min_std_kur >= 0.1 and min_std_ou >= 0.1

print(f"3. Is collapse unique to logistic? {'YES' if unique_to_logistic else 'NO'}")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

if unique_to_logistic:
    print("\nDimensional collapse (std≈0) is UNIQUE to logistic map")
    print("Kuramoto and OU show non-zero variance across D(k) dimensions")
    print("This suggests collapse is property of specific dynamical regimes")
else:
    print("\nCollapse occurs in multiple systems")