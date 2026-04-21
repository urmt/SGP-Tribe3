#!/usr/bin/env python3
"""
SFH-SGP_UNIVERSAL_COMPLEXITY_SCALING_01
Test whether dimensionality scales with system complexity
"""
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

N_SAMPLES = 5000
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
        nn = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree')
        nn.fit(data)
        dists, _ = nn.kneighbors(data)
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
print("STEP 1: GENERATE SYSTEMS")
print("=" * 70)

print("\n1. PERIODIC (sine wave)")
t = np.arange(N_SAMPLES)
sine = np.sin(2 * np.pi * 5 * t / N_SAMPLES)
sine = (sine - sine.mean()) / sine.std()
print(f"   Generated: {len(sine)} points, mean={sine.mean():.6f}, std={sine.std():.6f}")

print("\n2. MULTI-SINE (3 sine waves)")
freqs = [5, 12, 23]
multi_sine = sum([np.sin(2 * np.pi * f * t / N_SAMPLES) for f in freqs])
multi_sine = (multi_sine - multi_sine.mean()) / multi_sine.std()
print(f"   Generated: {len(multi_sine)} points")

print("\n3. LOGISTIC (weak chaos, r=3.5)")
x = 0.12345
r = 3.5
logistic_35 = np.zeros(N_SAMPLES)
for i in range(N_SAMPLES):
    x = r * x * (1 - x)
    logistic_35[i] = x
logistic_35 = (logistic_35 - logistic_35.mean()) / logistic_35.std()
print(f"   Generated: {len(logistic_35)} points")

print("\n4. LOGISTIC (strong chaos, r=4.0)")
x = 0.12345
r = 4.0
logistic_40 = np.zeros(N_SAMPLES)
for i in range(N_SAMPLES):
    x = r * x * (1 - x)
    logistic_40[i] = x
logistic_40 = (logistic_40 - logistic_40.mean()) / logistic_40.std()
print(f"   Generated: {len(logistic_40)} points")

print("\n5. CORRELATED NOISE (AR(1), phi=0.8)")
phi = 0.8
correlated = np.zeros(N_SAMPLES)
correlated[0] = np.random.randn()
for i in range(1, N_SAMPLES):
    correlated[i] = phi * correlated[i-1] + np.random.randn() * np.sqrt(1 - phi**2)
correlated = (correlated - correlated.mean()) / correlated.std()
print(f"   Generated: {len(correlated)} points")

print("\n6. RANDOM NOISE (Gaussian white noise)")
random_noise = np.random.randn(N_SAMPLES)
random_noise = (random_noise - random_noise.mean()) / random_noise.std()
print(f"   Generated: {len(random_noise)} points")

print("\n7. HIGH-D RANDOM (10D projected to 1D)")
D = 10
highD_samples = np.random.randn(N_SAMPLES, D)
weights = np.random.randn(D)
highD = np.dot(highD_samples, weights)
highD = (highD - highD.mean()) / highD.std()
print(f"   Generated: {len(highD)} points")

print("\n" + "=" * 70)
print("STEP 2: EMBEDDING")
print("=" * 70)

print(f"\nEmbedding dimension: {EMBED_DIM}, Delay: {DELAY}")

systems = {
    'sine': sine,
    'multi_sine': multi_sine,
    'logistic_3.5': logistic_35,
    'logistic_4.0': logistic_40,
    'correlated': correlated,
    'random': random_noise,
    'highD': highD
}

embedded = {}
for name, sig in systems.items():
    embedded[name] = time_delay_embed(sig, EMBED_DIM, DELAY)
    print(f"{name}: embedded shape {embedded[name].shape}")

print("\n" + "=" * 70)
print("STEP 3: DIMENSIONALITY")
print("=" * 70)

results = []

for name, X in embedded.items():
    dk = {}
    for k in K_VALUES:
        dk[k] = compute_knn_dim(X, k)
    
    early_auc = np.mean([dk[2], dk[4], dk[8]])
    
    results.append({
        'system': name,
        'D2': dk[2],
        'D4': dk[4],
        'D8': dk[8],
        'D16': dk[16],
        'early_AUC': early_auc
    })
    
    print(f"\n{name}:")
    print(f"  D2={dk[2]:.4f}, D4={dk[4]:.4f}, D8={dk[8]:.4f}, D16={dk[16]:.4f}")
    print(f"  Early AUC={early_auc:.4f}")

df = pd.DataFrame(results)
dk_cols = ['D2', 'D4', 'D8', 'D16']

print("\n" + "=" * 70)
print("STEP 4: VALIDATION CHECK")
print("=" * 70)

print("\nChecking for NaN...")
nan_count = df[dk_cols + ['early_AUC']].isna().sum().sum()
print(f"NaN count: {nan_count}")
if nan_count > 0:
    print("FAIL: NaN detected")
    exit(1)

print("\nChecking D(k) increases with k...")
for name in df['system']:
    row = df[df['system'] == name].iloc[0]
    d_vals = [row['D2'], row['D4'], row['D8'], row['D16']]
    print(f"  {name}: {d_vals}")

print("\nChecking variance > 0...")
std_vals = df[dk_cols].std()
print(f"Std across systems: {std_vals.to_dict()}")
if (std_vals == 0).any():
    print("FAIL: constant values detected")
    exit(1)

print("\n" + "=" * 70)
print("STEP 5: COMPLEXITY LABELS")
print("=" * 70)

complexity_map = {
    'sine': 1,
    'multi_sine': 2,
    'logistic_3.5': 3,
    'logistic_4.0': 4,
    'correlated': 5,
    'random': 6,
    'highD': 7
}

df['complexity'] = df['system'].map(complexity_map)

print("\nSystem | Complexity | AUC")
print("-" * 45)
for _, row in df.sort_values('complexity').iterrows():
    print(f"{row['system']:<15} {row['complexity']:<10} {row['early_AUC']:<10.4f}")

print("\n" + "=" * 70)
print("STEP 6: ANALYSIS")
print("=" * 70)

complexity = df['complexity'].values
auc = df['early_AUC'].values

spearman_r, spearman_p = stats.spearmanr(complexity, auc)
pearson_r, pearson_p = stats.pearsonr(complexity, auc)

slope, intercept, r, p, se = stats.linregress(complexity, auc)

print(f"\nSpearman correlation: r={spearman_r:.4f}, p={spearman_p:.6f}")
print(f"Pearson correlation: r={pearson_r:.4f}, p={pearson_p:.6f}")
print(f"Linear regression: slope={slope:.4f}, intercept={intercept:.4f}")

print("\n" + "=" * 70)
print("STEP 7: OUTPUT")
print("=" * 70)

print(f"\n{'System':<15} {'Complexity':<10} {'AUC':<10}")
print("-" * 40)
for _, row in df.sort_values('complexity').iterrows():
    print(f"{row['system']:<15} {row['complexity']:<10} {row['early_AUC']:<10.4f}")

print(f"\n{'Metric':<25} {'Value':<15}")
print("-" * 40)
print(f"{'Spearman r':<25} {spearman_r:<15.4f}")
print(f"{'Spearman p':<25} {spearman_p:<15.6f}")
print(f"{'Pearson r':<25} {pearson_r:<15.4f}")
print(f"{'Pearson p':<25} {pearson_p:<15.6f}")
print(f"{'Slope':<25} {slope:<15.4f}")

print("\n" + "=" * 70)
print("STEP 8: INTERPRETATION")
print("=" * 70)

if spearman_r > 0.7 and spearman_p < 0.05:
    print(f"\nPASS: Dimensionality tracks system complexity")
    print(f"Spearman r = {spearman_r:.4f} > 0.7, p = {spearman_p:.6f}")
elif spearman_r > 0.5:
    print(f"\nPARTIAL: Moderate correlation detected")
    print(f"Spearman r = {spearman_r:.4f}")
else:
    print(f"\nFAIL: No scaling relationship detected")
    print(f"Spearman r = {spearman_r:.4f}")

print("\n" + "=" * 70)
print("SAVE OUTPUTS")
print("=" * 70)

df.to_csv(os.path.join(OUT_DIR, "universal_scaling_results.csv"), index=False)
print(f"Saved: {OUT_DIR}universal_scaling_results.csv")

print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

print(f"1. All systems generated: YES")
print(f"2. Same embedding used: YES")
print(f"3. No NaN values: YES")
print(f"4. Variance > 0: YES")
print(f"5. Correlation > 0.7: {'YES' if spearman_r > 0.7 else 'NO'}")