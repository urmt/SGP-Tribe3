#!/usr/bin/env python3
"""
SFH-SGP_TRUE_TRAJECTORY_01
Compute TRUE trajectory in D(k) feature space
"""
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
print("STEP 1-2: SIMULATION AND D(k) COMPUTATION")
print("=" * 70)

N_SAMPLES = 5000
TRANSIENT = 1000
R_MIN, R_MAX, R_STEP = 3.0, 4.0, 0.02

r_values = np.round(np.arange(R_MIN, R_MAX + R_STEP, R_STEP), 2)
print(f"\nr range: {R_MIN} to {R_MAX}, step={R_STEP}")

all_dk_vectors = []

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
    
    all_dk_vectors.append({
        'r': r,
        'D2': dk[2],
        'D4': dk[4],
        'D8': dk[8],
        'D16': dk[16]
    })

df_dk = pd.DataFrame(all_dk_vectors).dropna()

print(f"\nGenerated: {len(df_dk)} D(k) vectors")

print("\n" + "=" * 70)
print("STEP 8: VALIDATION - D(k) VECTORS")
print("=" * 70)

print("\nExample D(k) vectors:")
for idx in [0, 25, 49]:
    if idx < len(df_dk):
        row = df_dk.iloc[idx]
        print(f"r={row['r']:.2f}: D2={row['D2']:.3f}, D4={row['D4']:.3f}, D8={row['D8']:.3f}, D16={row['D16']:.3f}")

dk_cols = ['D2', 'D4', 'D8', 'D16']
for col in dk_cols:
    std = df_dk[col].std()
    print(f"  {col} std: {std:.4f}")

print("\n" + "=" * 70)
print("STEP 3: FEATURE EXTRACTION FROM D(k) VECTORS")
print("=" * 70)

def extract_features_from_dk(dk_series):
    """Extract features from a single D(k) series at one r value"""
    d_vals = np.array([dk_series['D2'], dk_series['D4'], dk_series['D8'], dk_series['D16']])
    k_vals = np.array(K_VALUES)
    
    first_deriv = np.gradient(d_vals, k_vals)
    second_deriv = np.gradient(first_deriv, k_vals)
    
    return {
        'max_derivative': np.max(np.abs(first_deriv)),
        'max_jump': np.max(np.abs(np.diff(d_vals))),
        'baseline_std': np.std(d_vals),
        'curvature': np.max(np.abs(second_deriv))
    }

feature_list = []
for idx, row in df_dk.iterrows():
    feats = extract_features_from_dk(row)
    feats['r'] = row['r']
    feature_list.append(feats)

df_features = pd.DataFrame(feature_list)

print("\nFeatures computed from D(k) vectors:")
print(f"Shape: {df_features.shape}")

for col in ['max_derivative', 'max_jump', 'baseline_std', 'curvature']:
    std_val = df_features[col].std()
    print(f"  {col} std across r: {std_val:.6f}")

print("\n" + "=" * 70)
print("STEP 4: FEATURE MATRIX")
print("=" * 70)

feature_cols = ['max_derivative', 'max_jump', 'baseline_std', 'curvature']
X = df_features[feature_cols].values
print(f"Feature matrix: {X.shape}")

print("\n" + "=" * 70)
print("STEP 5: NORMALIZATION")
print("=" * 70)

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

print("Normalized features (first 3 rows):")
print(X_norm[:3])

print("\n" + "=" * 70)
print("STEP 6: PCA")
print("=" * 70)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_norm)

df_features['PC1'] = X_pca[:, 0]
df_features['PC2'] = X_pca[:, 1]

print(f"PCA variance explained: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")

print("\n" + "=" * 70)
print("STEP 7: TRAJECTORY ANALYSIS")
print("=" * 70)

r_vals = df_features['r'].values
pc1_vals = df_features['PC1'].values
pc2_vals = df_features['PC2'].values

corr_pc1_r, p1 = stats.spearmanr(r_vals, pc1_vals)
corr_pc2_r, p2 = stats.spearmanr(r_vals, pc2_vals)

first_deriv_pc1 = np.gradient(pc1_vals, r_vals)
max_deriv_idx = np.argmax(np.abs(first_deriv_pc1))
max_deriv_r = r_vals[max_deriv_idx]

print(f"\nPC1 vs r: r={corr_pc1_r:.4f}, p={p1:.6f}")
print(f"PC2 vs r: r={corr_pc2_r:.4f}, p={p2:.6f}")
print(f"Max PC1 slope at r = {max_deriv_r:.2f}")

mean_before = np.mean(first_deriv_pc1[:max_deriv_idx]) if max_deriv_idx > 0 else 0
mean_after = np.mean(first_deriv_pc1[max_deriv_idx:]) if max_deriv_idx < len(first_deriv_pc1) else 0

smooth_path = abs(corr_pc1_r) > 0.7
sharp_bend = abs(mean_after - mean_before) > 0.5

print("\n" + "=" * 70)
print("FINAL OUTPUT")
print("=" * 70)

print(f"\nr | PC1 | PC2")
print("-" * 35)
for i in range(0, len(df_features), 10):
    row = df_features.iloc[i]
    print(f"{row['r']:.2f} | {row['PC1']:.4f} | {row['PC2']:.4f}")

print("\n" + "=" * 70)
print("FINAL QUESTIONS")
print("=" * 70)

print(f"\n1. Continuous path in PCA space? {'YES' if smooth_path else 'NO'}")
print(f"2. Geometric bend near chaos? {'YES' if sharp_bend else 'NO'}")
print(f"3. PC1 tracks transition? {'YES' if corr_pc1_r > 0 else 'NO'}")

df_features.to_csv(os.path.join(OUT_DIR, "true_trajectory.csv"), index=False)
print(f"\nSaved: {OUT_DIR}true_trajectory.csv")