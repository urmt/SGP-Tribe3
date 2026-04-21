#!/usr/bin/env python3
"""
SFH-SGP_INVARIANT_DIMENSION_PROXY_01
Compute robust scaling dimension proxy
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

def lyapunov_exponent(r, x0=0.5, n=3000):
    x = x0
    lambdas = []
    for i in range(n):
        x = r * x * (1 - x)
        if i >= 100:
            df = r * (1 - 2 * x)
            if abs(df) > 0:
                lambdas.append(np.log(abs(df)))
    return np.mean(lambdas) if lambdas else 0

def dimension_proxy(embedded, n_points=300, n_neighbors=10):
    try:
        if len(embedded) > n_points:
            idx = np.random.choice(len(embedded), n_points, replace=False)
            data = embedded[idx]
        else:
            data = embedded
        
        nn = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree')
        nn.fit(data)
        dists, _ = nn.kneighbors(data)
        
        r_ij = dists[:, 1:].flatten()
        r_ij = r_ij[r_ij > 1e-6]
        
        if len(r_ij) < 50:
            return np.nan
        
        r_min, r_max = np.percentile(r_ij, [5, 95])
        
        counts = []
        radii = np.linspace(r_min, r_max, 25)
        
        for rad in radii:
            count = np.sum(r_ij < rad)
            counts.append(count / len(r_ij))
        
        counts = np.array(counts)
        counts = counts[counts > 0]
        radii = radii[:len(counts)]
        
        if len(counts) < 5:
            return np.nan
        
        log_r = np.log(radii)
        log_c = np.log(counts + 1e-10)
        
        valid = np.isfinite(log_r) & np.isfinite(log_c)
        if valid.sum() < 3:
            return np.nan
        
        valid_r = log_r[valid]
        valid_c = log_c[valid]
        
        slope, _ = np.polyfit(valid_r, valid_c, 1)
        
        return slope
        
    except:
        return np.nan

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 80)
print("DIMENSION PROXY ANALYSIS")
print("=" * 80)

N_SAMPLES = 5000
TRANSIENT = 1000
R_MIN, R_MAX, R_STEP = 3.0, 4.0, 0.02

r_values = np.round(np.arange(R_MIN, R_MAX + R_STEP, R_STEP), 2)
print(f"\nr range: {R_MIN} to {R_MAX}, step={R_STEP}")

results = []

for idx, r in enumerate(r_values):
    if idx % 10 == 0:
        print(f"  Processing r={r:.2f}...")
    
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
    
    lam = lyapunov_exponent(r)
    
    d_proxy = dimension_proxy(embedded, n_points=300)
    
    results.append({
        'r': r,
        'AUC': early_auc,
        'std_Dk': dk_std,
        'lambda': lam,
        'D_proxy': d_proxy
    })

df = pd.DataFrame(results).dropna()

print(f"\nGenerated: {len(df)} valid points")

print("\n" + "=" * 80)
print("SAMPLE OUTPUT")
print("=" * 80)

print(f"\n{'r':<8} {'AUC':<14} {'std_Dk':<12} {'lambda':<12} {'D_proxy':<12}")
print("-" * 65)
for _, row in df[df['r'].isin([3.2, 3.5, 3.7, 3.9])].iterrows():
    dp = row['D_proxy'] if not np.isnan(row['D_proxy']) else -999
    print(f"{row['r']:<8.2f} {row['AUC']:<14.4f} {row['std_Dk']:<12.4f} {row['lambda']:<12.4f} {dp:<12.4f}")

print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

d_proxy_std = df['D_proxy'].std()
d_proxy_mean = df['D_proxy'].mean()

print(f"\nD_proxy stats: mean={d_proxy_mean:.4f}, std={d_proxy_std:.4f}")
print(f"D_proxy range: {df['D_proxy'].min():.4f} - {df['D_proxy'].max():.4f}")

if d_proxy_std < 0.001:
    print("\nFAIL: D_proxy is constant!")
    exit(1)

print("\nValidation: PASSED")

print("\n" + "=" * 80)
print("CORRELATIONS")
print("=" * 80)

pearson_auc_lambda, p1 = stats.pearsonr(df['AUC'], df['lambda'])
spearman_auc_lambda, sp1 = stats.spearmanr(df['AUC'], df['lambda'])

pearson_auc_dp, p2 = stats.pearsonr(df['AUC'], df['D_proxy'])
spearman_auc_dp, sp2 = stats.spearmanr(df['AUC'], df['D_proxy'])

print(f"\nAUC vs λ: Pearson r={pearson_auc_lambda:.4f}, Spearman r={spearman_auc_lambda:.4f}")
print(f"AUC vs D_proxy: Pearson r={pearson_auc_dp:.4f}, Spearman r={spearman_auc_dp:.4f}")

print("\n" + "=" * 80)
print("FINAL QUESTIONS")
print("=" * 80)

periodic = df[df['lambda'] < 0]
chaotic = df[df['lambda'] > 0]
collapse_only_periodic = periodic['std_Dk'].mean() < 0.1 and chaotic['std_Dk'].mean() > 0.1

print(f"\n1. AUC vs λ correlation: {'YES' if abs(pearson_auc_lambda) > 0.5 else 'NO'}")
print(f"2. AUC vs D_proxy correlation: {'YES' if abs(pearson_auc_dp) > 0.3 else 'NO'}")
print(f"3. Collapse only λ<0: {'YES' if collapse_only_periodic else 'NO'}")
print(f"4. D(k) linked to instability AND geometry: {'YES' if abs(pearson_auc_lambda) > 0.5 and abs(pearson_auc_dp) > 0.3 else 'PARTIAL'}")

df.to_csv(os.path.join(OUT_DIR, "invariant_matching.csv"), index=False)
print(f"\nSaved: {OUT_DIR}invariant_matching.csv")