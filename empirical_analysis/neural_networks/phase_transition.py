#!/usr/bin/env python3
"""
SFH-SGP_PHASE_TRANSITION_ZOOM_01
Determine whether dimensionality transition near chaos onset is sharp or continuous
"""
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

N_SAMPLES = 5000
TRANSIENT = 1000
EMBED_DIM = 10
DELAY = 2
K_VALUES = [2, 4, 8, 16]
R_MIN = 3.54
R_MAX = 3.60
R_STEP = 0.002
OUT_DIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/"

np.random.seed(42)

def compute_knn_dim(data, k):
    try:
        n, d = data.shape
        if n <= k + 1 or d < 2:
            return np.nan
        if d < 5:
            data = np.hstack([data, np.zeros((n, 5-d))])
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
print("STEP 1 & 2: PARAMETER RANGE & DATA GENERATION")
print("=" * 70)

r_values = np.arange(R_MIN, R_MAX + R_STEP, R_STEP)
r_values = np.round(r_values, 3)

print(f"\nr range: {R_MIN} to {R_MAX}, step={R_STEP}")
print(f"Number of points: {len(r_values)}")

all_results = []

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
    
    all_results.append({
        'r': r,
        'D2': dk[2],
        'D4': dk[4],
        'D8': dk[8],
        'D16': dk[16],
        'early_AUC': early_auc
    })

df = pd.DataFrame(all_results)
dk_cols = ['D2', 'D4', 'D8', 'D16']

df = df.dropna()
print(f"Valid data points: {len(df)}")

print("\n" + "=" * 70)
print("STEP 3-4: EMBEDDING & DIMENSIONALITY")
print("=" * 70)

print("\nSample points:")
for idx in [0, len(df)//4, len(df)//2, 3*len(df)//4, len(df)-1]:
    if idx < len(df):
        row = df.iloc[idx]
        print(f"r={row['r']:.3f}: D2={row['D2']:.2f}, D4={row['D4']:.2f}, D8={row['D8']:.2f}, AUC={row['early_AUC']:.2f}")

print("\n" + "=" * 70)
print("STEP 5: VALIDATION")
print("=" * 70)

nan_count = df[dk_cols + ['early_AUC']].isna().sum().sum()
print(f"NaN count: {nan_count}")
if nan_count > 0:
    print("WARNING: NaN values filtered")

std_vals = df[dk_cols].std()
print(f"Std across r: {std_vals.to_dict()}")

if (std_vals == 0).all():
    print("FAIL: constant values")
    exit(1)

print("\n" + "=" * 70)
print("STEP 6: OUTPUT TABLE")
print("=" * 70)

print(f"\n{'r':<8} {'AUC':<12}")
print("-" * 20)
for i in range(0, len(df), 5):
    row = df.iloc[i]
    print(f"{row['r']:<8.3f} {row['early_AUC']:<12.4f}")
print("...")
for i in range(len(df)-5, len(df)):
    row = df.iloc[i]
    print(f"{row['r']:<8.3f} {row['early_AUC']:<12.4f}")

print("\n" + "=" * 70)
print("STEP 7: ANALYSIS")
print("=" * 70)

r_vals = df['r'].values
auc_vals = df['early_AUC'].values

spearman_r, spearman_p = stats.spearmanr(r_vals, auc_vals)
pearson_r, pearson_p = stats.pearsonr(r_vals, auc_vals)

df['derivative'] = np.gradient(auc_vals, r_vals)

mean_deriv = df['derivative'].mean()
max_deriv_idx = df['derivative'].abs().idxmax()
max_deriv_r = df.loc[max_deriv_idx, 'r']
max_deriv_val = df.loc[max_deriv_idx, 'derivative']

auc_range = auc_vals.max() - auc_vals.min()

print(f"\nSpearman correlation: r={spearman_r:.4f}, p={spearman_p:.6f}")
print(f"Pearson correlation: r={pearson_r:.4f}, p={pearson_p:.6f}")
print(f"\nDerivative analysis:")
print(f"  Mean |dAUC/dr|: {abs(mean_deriv):.4f}")
print(f"  Max |dAUC/dr| location: r = {max_deriv_r:.3f}")
print(f"  Max |dAUC/dr| value: {abs(max_deriv_val):.4f}")
print(f"  AUC range: {auc_range:.4f}")

max_jump_idx = df['early_AUC'].diff().abs().idxmax()
max_jump = df.loc[max_jump_idx, 'early_AUC'] - df.loc[max_jump_idx-1, 'early_AUC'] if max_jump_idx > 0 else 0
max_jump_r = df.loc[max_jump_idx, 'r'] if max_jump_idx > 0 else df.iloc[0]['r']

print(f"\nMax single-step jump:")
print(f"  At r = {max_jump_r:.3f}")
print(f"  Magnitude = {abs(max_jump):.4f}")

print("\n" + "=" * 70)
print("STEP 8: INTERPRETATION")
print("=" * 70)

if abs(max_jump) > 5.0:
    print(f"\nDISCONTINUOUS TRANSITION detected")
    print(f"Sharp jump of {abs(max_jump):.2f} at r = {max_jump_r:.3f}")
else:
    print(f"\nCONTINUOUS TRANSITION detected")
    print(f"AUC changes smoothly: max jump = {abs(max_jump):.4f} per step")

print("\n" + "=" * 70)
print("SAVE RESULTS")
print("=" * 70)

df.to_csv(os.path.join(OUT_DIR, "phase_transition_results.csv"), index=False)
print(f"Saved: {OUT_DIR}phase_transition_results.csv")

print("\n" + "=" * 70)
print("KEY FINDING")
print("=" * 70)

if abs(max_jump) > 5.0:
    print(f"\nThe dimensionality transition at the chaos threshold")
    print(f"is DISCONTINUOUS (abrupt jump)")
else:
    print(f"\nThe dimensionality transition at the chaos threshold")
    print(f"is CONTINUOUS (smooth change)")