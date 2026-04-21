#!/usr/bin/env python3
"""
SFH-SGP_INVARIANT_MATCHING_01
Test D(k) correlation with dynamical invariants
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

def lyapunov_exponent(r, x0=0.5, n=5000):
    """Compute Lyapunov exponent for logistic map"""
    x = x0
    lambdas = []
    for i in range(n):
        x = r * x * (1 - x)
        if i >= 100:
            df = r * (1 - 2 * x)
            if abs(df) > 0:
                lambdas.append(np.log(abs(df)))
    return np.mean(lambdas) if lambdas else 0

def correlation_dimension_simple(x, n_bins=20):
    """Simplified correlation dimension"""
    try:
        x_flat = x.flatten()
        if len(x_flat) < 50:
            return np.nan
        
        counts = []
        for i in range(n_bins):
            threshold = (i + 1) * 0.1
            nn = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
            nn.fit(x_flat.reshape(-1, 1))
            dists, _ = nn.kneighbors(x_flat.reshape(-1, 1))
            count = np.sum(dists[:, 1] < threshold)
            counts.append(count / len(x_flat))
        
        return np.mean(counts) * 10
    except:
        return np.nan

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 80)
print("LOGISTIC MAP INVARIANT ANALYSIS")
print("=" * 80)

N_SAMPLES = 5000
TRANSIENT = 1000
R_MIN, R_MAX, R_STEP = 3.0, 4.0, 0.02

r_values = np.round(np.arange(R_MIN, R_MAX + R_STEP, R_STEP), 2)
print(f"\nr range: {R_MIN} to {R_MAX}, step={R_STEP}")

results = []

for idx, r in enumerate(r_values):
    if idx % 5 == 0:
        print(f"  Processing r={r:.2f}...", end=" ")
    
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
    dk_std = np.std([dk[2], dk[4], dk[8], dk[16]])
    
    lam = lyapunov_exponent(r, n=3000)
    
    D2 = correlation_dimension_simple(embedded[::10])
    
    results.append({
        'r': r,
        'AUC': early_auc,
        'std_Dk': dk_std,
        'lambda': lam,
        'D2': D2
    })
    
    if idx % 5 == 0:
        print(f"AUC={early_auc:.2f}, λ={lam:.3f}")

df = pd.DataFrame(results)
df = df.dropna()

print(f"\nGenerated: {len(df)} valid points")

print("\n" + "=" * 80)
print("SAMPLE OUTPUT")
print("=" * 80)

print(f"\n{'r':<8} {'AUC':<14} {'std_Dk':<12} {'lambda':<12} {'D2':<10}")
print("-" * 60)
for _, row in df[df['r'].isin([3.2, 3.5, 3.7, 3.9])].iterrows():
    print(f"{row['r']:<8.2f} {row['AUC']:<14.4f} {row['std_Dk']:<12.4f} {row['lambda']:<12.4f} {row['D2']:<10.4f}")

print("\n" + "=" * 80)
print("CORRELATIONS")
print("=" * 80)

pearson_auc_lambda, p1 = stats.pearsonr(df['AUC'], df['lambda'])
spearman_auc_lambda, sp1 = stats.spearmanr(df['AUC'], df['lambda'])

pearson_auc_D2, p2 = stats.pearsonr(df['AUC'], df['D2'])
spearman_auc_D2, sp2 = stats.spearmanr(df['AUC'], df['D2'])

print(f"\nAUC vs λ: Pearson r={pearson_auc_lambda:.4f}, Spearman r={spearman_auc_lambda:.4f}")
print(f"AUC vs D2: Pearson r={pearson_auc_D2:.4f}, Spearman r={spearman_auc_D2:.4f}")

print("\n" + "=" * 80)
print("REGIME CHECK")
print("=" * 80)

periodic = df[df['lambda'] < 0]
chaotic = df[df['lambda'] > 0]

print(f"\nPeriodic: {len(periodic)} points, mean std(Dk) = {periodic['std_Dk'].mean():.4f}")
print(f"Chaotic: {len(chaotic)} points, mean std(Dk) = {chaotic['std_Dk'].mean():.4f}")

collapse_only_periodic = periodic['std_Dk'].mean() < 0.1 and chaotic['std_Dk'].mean() > 0.1

print(f"Collapse only periodic: {collapse_only_periodic}")

print("\n" + "=" * 80)
print("FINAL QUESTIONS")
print("=" * 80)

print(f"\n1. AUC vs λ: {'YES' if abs(pearson_auc_lambda) > 0.5 else 'NO'}")
print(f"2. AUC vs D2: {'YES' if abs(pearson_auc_D2) > 0.5 else 'NO'}")
print(f"3. Collapse only λ<0: {'YES' if collapse_only_periodic else 'NO'}")
print(f"4. D(k) as proxy: {'YES' if abs(pearson_auc_lambda) > 0.5 else 'PARTIAL'}")

df.to_csv(os.path.join(OUT_DIR, "invariant_matching.csv"), index=False)
print(f"\nSaved: {OUT_DIR}invariant_matching.csv")