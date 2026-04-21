#!/usr/bin/env python3
"""
SFH-SGP_TRANSITION_TRAJECTORY_SIMPLE
Simpler trajectory analysis using AUC directly
"""
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy import stats
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
print("LOGISTIC MAP TRAJECTORY")
print("=" * 70)

N_SAMPLES = 5000
TRANSIENT = 1000
R_MIN, R_MAX, R_STEP = 3.0, 4.0, 0.02

r_values = np.round(np.arange(R_MIN, R_MAX + R_STEP, R_STEP), 2)
print(f"\nr range: {R_MIN} to {R_MAX}, step={R_STEP}")

results = []
for r in r_values:
    x = 0.5
    samples = np.zeros(N_SAMPLES + TRANSIENT)
    for i in range(N_SAMPLES + TRANSIENT):
        x = r * x * (1 - x)
        samples[i] = x
    
    signal = samples[TRANSIENT:]
    signal = (signal - signal.mean()) / (signal.std() + 1e-10)
    
    embedded = time_delay_embed(signal, EMBED_DIM, DELAY)
    
    dk = {}
    for k in K_VALUES:
        dk[k] = compute_knn_dim(embedded, k)
    
    early_auc = np.mean([dk[2], dk[4], dk[8]])
    
    results.append({'r': r, 'AUC': early_auc, 
                   'D2': dk[2], 'D4': dk[4], 'D8': dk[8], 'D16': dk[16]})

df = pd.DataFrame(results).dropna()
print(f"Generated: {len(df)} points")

print("\n" + "=" * 70)
print("TRAJECTORY ANALYSIS")
print("=" * 70)

r_vals = df['r'].values
auc_vals = df['AUC'].values

first_deriv = np.gradient(auc_vals, r_vals)
second_deriv = np.gradient(first_deriv, r_vals)

max_deriv_idx = np.argmax(np.abs(first_deriv))
max_deriv_r = r_vals[max_deriv_idx]

print(f"\nMax derivative at r = {max_deriv_r:.2f}")

smooth_regions = []
prev = 0
for i in range(len(r_vals)):
    if r_vals[i] < 3.55:
        smooth_regions.append(auc_vals[i])
    elif r_vals[i] > 3.57:
        if len(smooth_regions) == 0:
            smooth_regions = [auc_vals[i]]
        else:
            break

smooth_auc = np.std(auc_vals[:len(smooth_regions)]) if len(smooth_regions) > 1 else 0

transition_region = auc_vals[len(smooth_regions):len(smooth_regions)+5]
jump = np.max(np.abs(np.diff(transition_region))) if len(transition_region) > 1 else 0

print(f"Transition region jump: {jump:.2f}")

corr, p = stats.spearmanr(r_vals, auc_vals)
print(f"\nAUC vs r: r={corr:.4f}, p={p:.6f}")

monotonic = corr > 0.7
sharp_bend = jump > 10

print("\n" + "=" * 70)
print("FINAL QUESTIONS")
print("=" * 70)

print(f"\n1. Smooth path? {'YES' if monotonic else 'NO'} (r={corr:.2f})")
print(f"2. Sharp bend? {'YES' if sharp_bend else 'NO'} (jump={jump:.2f})")
print(f"3. Tracks chaos? {'YES' if corr > 0 else 'NO'}")

df.to_csv(os.path.join(OUT_DIR, "logistic_trajectory.csv"), index=False)
print(f"\nSaved: {OUT_DIR}logistic_trajectory.csv")