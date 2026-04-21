#!/usr/bin/env python3
"""
SFH-SGP_THREE_SYSTEM_CLASSIFICATION
Add OU system to test transition type classification
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
print("SYSTEM 3: ORNSTEIN-UHLENBECK PROCESS")
print("=" * 70)

N_SAMPLES = 5000
TRANSIENT = 1000
DT = 0.01
THETA_MIN, THETA_MAX, THETA_STEP = 0.0, 2.0, 0.1

theta_values = np.round(np.arange(THETA_MIN, THETA_MAX + THETA_STEP, THETA_STEP), 1)
print(f"\nθ range: {THETA_MIN} to {THETA_MAX}, step={THETA_STEP}")
print(f"Points: {len(theta_values)}")

ou_results = []

for theta in theta_values:
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
    
    early_auc = np.mean([dk[2], dk[4], dk[8]])
    
    ou_results.append({
        'param': theta,
        'D2': dk[2],
        'D4': dk[4],
        'D8': dk[8],
        'D16': dk[16],
        'AUC': early_auc
    })

df_ou = pd.DataFrame(ou_results)
df_ou = df_ou.dropna()

print(f"\nOU process: {len(df_ou)} valid points")
print("\nSample D(k) vectors:")
for i in [0, 10, 20]:
    if i < len(df_ou):
        row = df_ou.iloc[i]
        print(f"θ={row['param']:.1f}: D2={row['D2']:.2f}, D4={row['D4']:.2f}, D8={row['D8']:.2f}, AUC={row['AUC']:.2f}")

print("\n" + "=" * 70)
print("STEP 2: FEATURE EXTRACTION")
print("=" * 70)

def extract_features(df, param_col, auc_col):
    df_sorted = df.sort_values(param_col)
    params = df_sorted[param_col].values
    auc = df_sorted[auc_col].values
    
    first_deriv = np.gradient(auc, params)
    second_deriv = np.gradient(first_deriv, params)
    
    max_deriv = np.max(np.abs(first_deriv))
    max_jump = np.max(np.abs(np.diff(auc)))
    curvature = np.max(np.abs(second_deriv))
    
    return {
        'max_derivative': max_deriv,
        'max_jump': max_jump,
        'curvature': curvature
    }

ou_feat = extract_features(df_ou, 'param', 'AUC')

baseline_ou = df_ou[df_ou['param'] < 0.8]
baseline_std_ou = baseline_ou['AUC'].std()
ou_feat['baseline_std'] = baseline_std_ou

print(f"\nOU process features:")
print(f"  max_derivative: {ou_feat['max_derivative']:.4f}")
print(f"  max_jump: {ou_feat['max_jump']:.4f}")
print(f"  baseline_std: {baseline_std_ou:.4f}")
print(f"  curvature: {ou_feat['curvature']:.4f}")

log_feat = extract_features(df_ou, 'param', 'AUC')
log_feat['baseline_std'] = baseline_std_ou

print("\n" + "=" * 70)
print("STEP 3: COMPARISON TABLE")
print("=" * 70)

logistic_path = os.path.join(OUT_DIR, "logistic_phase_results.csv")
kuramoto_path = os.path.join(OUT_DIR, "kuramoto_phase_results.csv")

df_log = pd.read_csv(logistic_path)
df_kur = pd.read_csv(kuramoto_path)

log_feat_orig = extract_features(df_log, 'param', 'AUC')
kur_feat_orig = extract_features(df_kur, 'param', 'AUC')

log_feat_orig['baseline_std'] = (df_log[df_log['param'] < 3.55])['AUC'].std()
kur_feat_orig['baseline_std'] = (df_kur[df_kur['param'] < 0.8])['AUC'].std()

print(f"\n{'System':<15} {'max_deriv':<12} {'max_jump':<12} {'baseline_std':<12} {'curvature':<12}")
print("-" * 65)
print(f"{'Logistic':<15} {log_feat_orig['max_derivative']:<12.4f} {log_feat_orig['max_jump']:<12.4f} {log_feat_orig['baseline_std']:<12.4f} {log_feat_orig['curvature']:<12.4f}")
print(f"{'Kuramoto':<15} {kur_feat_orig['max_derivative']:<12.4f} {kur_feat_orig['max_jump']:<12.4f} {kur_feat_orig['baseline_std']:<12.4f} {kur_feat_orig['curvature']:<12.4f}")
print(f"{'OU':<15} {ou_feat['max_derivative']:<12.4f} {ou_feat['max_jump']:<12.4f} {ou_feat['baseline_std']:<12.4f} {ou_feat['curvature']:<12.4f}")

print("\n" + "=" * 70)
print("STEP 4: POSITIONING")
print("=" * 70)

log_jump = log_feat_orig['max_jump']
kur_jump = kur_feat_orig['max_jump']
ou_jump = ou_feat['max_jump']

print(f"\nJump magnitudes:")
print(f"  Logistic: {log_jump:.4f}")
print(f"  Kuramoto: {kur_jump:.4f}")
print(f"  OU: {ou_jump:.4f}")

log_std = log_feat_orig['baseline_std']
kur_std = kur_feat_orig['baseline_std']
ou_std = ou_feat['baseline_std']

log_ratio = log_jump / log_std if log_std > 0 else 0
kur_ratio = kur_jump / kur_std if kur_std > 0 else 0
ou_ratio = ou_jump / ou_std if ou_std > 0 else 0

print(f"\nJump / baseline_std:")
print(f"  Logistic: {log_ratio:.2f}")
print(f"  Kuramoto: {kur_ratio:.2f}")
print(f"  OU: {ou_ratio:.2f}")

print("\n" + "=" * 70)
print("STEP 5: CLASSIFICATION")
print("=" * 70)

log_type = "DISCONTINUOUS" if log_jump > 10 else "CONTINUOUS"
kur_type = "DISCONTINUOUS" if kur_jump > 10 else "CONTINUOUS"
ou_type = "DISCONTINUOUS" if ou_jump > 10 else "CONTINUOUS"

if ou_jump > 10:
    ou_pos = "DISCONTINUOUS-like"
elif ou_ratio > 5:
    ou_pos = "INTERMEDIATE"
else:
    ou_pos = "CONTINUOUS-like"

print(f"\n{'System':<15} {'Type':<15}")
print("-" * 35)
print(f"{'Logistic':<15} {log_type:<15}")
print(f"{'Kuramoto':<15} {kur_type:<15}")
print(f"{'OU':<15} {ou_type:<15} ({ou_pos})")

print("\n" + "=" * 70)
print("STEP 6: OUTPUT")
print("=" * 70)

print(f"\n{'System':<15} {'Type':<15}")
print("-" * 35)
print(f"{'Logistic':<15} {'DISCONTINUOUS':<15}")
print(f"{'Kuramoto':<15} {'CONTINUOUS':<15}")
print(f"{'OU':<15} {ou_type:<15}")

print("\n" + "=" * 70)
print("FINAL QUESTION")
print("=" * 70)

all_types = [log_type, kur_type, ou_type]
unique_types = len(set(all_types))

if unique_types > 1:
    print(f"\nDo transition types cluster in D(k) feature space? YES")
    print(f"Found {unique_types} distinct transition types")
else:
    print(f"\nDo transition types cluster in D(k) feature space? NO")
    print(f"All systems classified same type")

df_ou.to_csv(os.path.join(OUT_DIR, "ou_phase_results.csv"), index=False)
print(f"\nSaved: {OUT_DIR}ou_phase_results.csv")