#!/usr/bin/env python3
"""
SFH-SGP_TRANSITION_TRAJECTORY_01
Test whether logistic map traces continuous path in D(k) feature space
"""
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
print("SYSTEM: LOGISTIC MAP TRAJECTORY")
print("=" * 70)

N_SAMPLES = 5000
TRANSIENT = 1000
R_MIN, R_MAX, R_STEP = 3.0, 4.0, 0.02

r_values = np.round(np.arange(R_MIN, R_MAX + R_STEP, R_STEP), 2)
print(f"\nr range: {R_MIN} to {R_MAX}, step={R_STEP}")
print(f"Points: {len(r_values)}")

trajectory_results = []

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
    
    trajectory_results.append({
        'r': r,
        'D2': dk[2],
        'D4': dk[4],
        'D8': dk[8],
        'D16': dk[16],
        'AUC': early_auc
    })

df_traj = pd.DataFrame(trajectory_results)
df_traj = df_traj.dropna()

print(f"\nGenerated: {len(df_traj)} points")

print("\n" + "=" * 70)
print("STEP 2-3: FEATURE EXTRACTION & PCA")
print("=" * 70)

r_vals = df_traj['r'].values
auc_vals = df_traj['AUC'].values

first_deriv = np.gradient(auc_vals, r_vals)
second_deriv = np.gradient(first_deriv, r_vals)

df_traj['first_deriv'] = first_deriv
df_traj['second_deriv'] = second_deriv

feature_matrix = np.column_stack([
    np.abs(first_deriv),
    np.abs(np.diff(auc_vals, prepend=auc_vals[0])),
    np.ones(len(r_vals)) * np.std(auc_vals),
    np.abs(second_deriv)
])

scaler = StandardScaler()
X_norm = scaler.fit_transform(feature_matrix)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_norm)

df_traj['PC1'] = X_pca[:, 0]
df_traj['PC2'] = X_pca[:, 1]

print(f"PCA variance: {pca.explained_variance_ratio_}")

print("\n" + "=" * 70)
print("STEP 4: TRAJECTORY ANALYSIS")
print("=" * 70)

print(f"\nr | PC1 | PC2")
print("-" * 40)
for i in range(0, len(df_traj), 10):
    row = df_traj.iloc[i]
    print(f"{row['r']:<6.2f} {row['PC1']:<8.4f} {row['PC2']:<8.4f}")

print("\n" + "=" * 70)
print("STEP 5: STRUCTURE TEST")
print("=" * 70)

corr, p = stats.spearmanr(r_vals, df_traj['PC1'].values)

max_deriv_idx = np.argmax(np.abs(first_deriv))
max_deriv_r = r_vals[max_deriv_idx]

mean_before = np.mean(first_deriv[:max_deriv_idx]) if max_deriv_idx > 0 else 0
mean_after = np.mean(first_deriv[max_deriv_idx:]) if max_deriv_idx < len(first_deriv) else 0

monotonic = corr > 0.8
sharp_change = abs(mean_after - mean_before) > 1.0

print(f"\nPC1 vs r: r={corr:.4f}, p={p:.4f}")
print(f"Max slope at r = {max_deriv_r:.2f}")
print(f"Mean slope before: {mean_before:.4f}, after: {mean_after:.4f}")

print("\n" + "=" * 70)
print("FINAL QUESTIONS")
print("=" * 70)

print(f"\n1. Does trajectory form a smooth path? {'YES' if monotonic else 'NO'}")
print(f"2. Is there a sharp bend near transition? {'YES' if sharp_change else 'NO'}")
print(f"3. Does PC1 track increasing chaos? {'YES' if corr > 0 else 'NO'}")

df_traj.to_csv(os.path.join(OUT_DIR, "logistic_trajectory.csv"), index=False)
print(f"\nSaved: {OUT_DIR}logistic_trajectory.csv")