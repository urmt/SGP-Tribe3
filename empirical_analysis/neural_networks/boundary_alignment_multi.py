#!/usr/bin/env python3
"""
SFH-SGP_BOUNDARY_ALIGNMENT_MULTI
Test D(k) collapse alignment across multiple dynamical systems:
1. Logistic map (chaos boundary)
2. Kuramoto (synchronization boundary)
3. OU process (no boundary - continuous)
"""
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

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


def lyapunov_logistic(r, x0=0.5, n=5000):
    x = x0
    lambdas = []
    for i in range(n):
        x = r * x * (1 - x)
        if i >= 100:
            deriv = abs(r * (1 - 2 * x))
            if deriv > 0:
                lambdas.append(np.log(deriv))
    return np.mean(lambdas) if lambdas else 0


def order_parameter_kuramoto(theta, K):
    n = len(theta)
    r = np.abs(np.sum(np.exp(1j * theta)) / n)
    return r


def kuramoto_synchronization(K_range, n_oscillators=200, n_steps=5000, burn=1000):
    results = []
    
    for K in K_range:
        theta = np.random.uniform(0, 2*np.pi, n_oscillators)
        
        for t in range(n_steps):
            d_theta = K * np.sin(theta - theta.reshape(-1, 1))
            d_theta = d_theta.mean(axis=1)
            theta = theta + d_theta
        
        r = order_parameter_kuramoto(theta, K)
        results.append({'K': K, 'r_order': r})
    
    return pd.DataFrame(results)


def ou_process(theta, mu=0, sigma=1.0, dt=0.01, n_steps=5000):
    x = np.zeros(n_steps)
    x[0] = np.random.randn()
    
    for t in range(1, n_steps):
        x[t] = x[t-1] + theta * (mu - x[t-1]) * dt + sigma * np.sqrt(dt) * np.random.randn()
    
    return x


K_LIST = [2, 4, 8, 16]
COLLAPSE_THRESH = 0.05

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 80)
print("MULTI-SYSTEM BOUNDARY ALIGNMENT")
print("=" * 80)

results_all = []

print("\n" + "=" * 80)
print("1. LOGISTIC MAP (r = 3.54 to 3.60)")
print("=" * 80)

R_VALUES = np.round(np.arange(3.54, 3.61, 0.002), 3)
N_TOTAL, N_BURN = 6000, 1000
EMBED_DIM, DELAY = 10, 2

for r in R_VALUES:
    x = 0.5
    trajectory = np.zeros(N_TOTAL)
    for i in range(N_TOTAL):
        x = r * x * (1 - x)
        trajectory[i] = x
    
    signal = trajectory[N_BURN:]
    signal = (signal - signal.mean()) / (signal.std() + 1e-10)
    
    embedded = time_delay_embed(signal, EMBED_DIM, DELAY)
    
    dk_vals = [compute_knn_dim(embedded, k) for k in K_LIST]
    std_dk = np.std(dk_vals)
    auc = np.trapz(dk_vals, np.log(K_LIST))
    lam = lyapunov_logistic(r)
    
    collapse = std_dk < COLLAPSE_THRESH
    
    results_all.append({
        'system': 'LOGISTIC',
        'param': r,
        'lambda': lam,
        'D2': dk_vals[0],
        'D4': dk_vals[1],
        'D8': dk_vals[2],
        'D16': dk_vals[3],
        'std_Dk': std_dk,
        'AUC': auc,
        'collapse': collapse
    })

print(f"   Logistic: {len(R_VALUES)} points, collapse detected: {sum([r['collapse'] for r in results_all if r['system'] == 'LOGISTIC'])}")

print("\n" + "=" * 80)
print("2. KURAMOTO (K = 0.0 to 2.0)")
print("=" * 80)

KURAMOTO_K_VALUES = np.round(np.arange(0.0, 2.1, 0.1), 1)

for K in KURAMOTO_K_VALUES:
    n_oscillators = 200
    n_steps = 5000
    theta = np.random.uniform(0, 2*np.pi, n_oscillators)
    
    for t in range(n_steps):
        d_theta = K * np.sin(theta - theta.reshape(-1, 1))
        d_theta = d_theta.mean(axis=1)
        theta = theta + d_theta
    
    r_order = np.abs(np.sum(np.exp(1j * theta)) / n_oscillators)
    
    signal = theta
    signal = (signal - signal.mean()) / (signal.std() + 1e-10)
    
    embedded = time_delay_embed(signal, EMBED_DIM, DELAY)
    
    dk_vals = [compute_knn_dim(embedded, k) for k in K_LIST]
    std_dk = np.std(dk_vals)
    auc = np.trapz(dk_vals, np.log(K_LIST))
    
    collapse = std_dk < COLLAPSE_THRESH
    
    results_all.append({
        'system': 'KURAMOTO',
        'param': K,
        'r_order': r_order,
        'D2': dk_vals[0],
        'D4': dk_vals[1],
        'D8': dk_vals[2],
        'D16': dk_vals[3],
        'std_Dk': std_dk,
        'AUC': auc,
        'collapse': collapse
    })

print(f"   Kuramoto: {len(KURAMOTO_K_VALUES)} points, collapse detected: {sum([r['collapse'] for r in results_all if r['system'] == 'KURAMOTO'])}")

print("\n" + "=" * 80)
print("3. OU PROCESS (theta = 0.1 to 5.0)")
print("=" * 80)

OU_THETA_VALUES = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

for theta in OU_THETA_VALUES:
    signal = ou_process(theta, mu=0, sigma=1.0, dt=0.01, n_steps=5000)
    signal = (signal - signal.mean()) / (signal.std() + 1e-10)
    
    embedded = time_delay_embed(signal, EMBED_DIM, DELAY)
    
    dk_vals = [compute_knn_dim(embedded, k) for k in K_LIST]
    std_dk = np.std(dk_vals)
    auc = np.trapz(dk_vals, np.log(K_LIST))
    
    collapse = std_dk < COLLAPSE_THRESH
    
    results_all.append({
        'system': 'OU',
        'param': theta,
        'D2': dk_vals[0],
        'D4': dk_vals[1],
        'D8': dk_vals[2],
        'D16': dk_vals[3],
        'std_Dk': std_dk,
        'AUC': auc,
        'collapse': collapse
    })

print(f"   OU: {len(OU_THETA_VALUES)} points, collapse detected: {sum([r['collapse'] for r in results_all if r['system'] == 'OU'])}")

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

for system in ['LOGISTIC', 'KURAMOTO', 'OU']:
    df_sys = pd.DataFrame([r for r in results_all if r['system'] == system])
    if len(df_sys) > 0:
        print(f"\n{system}:")
        print(f"  Points: {len(df_sys)}")
        print(f"  Collapse: {df_sys['collapse'].sum()}")
        print(f"  std_Dk range: {df_sys['std_Dk'].min():.4f} - {df_sys['std_Dk'].max():.4f}")
        print(f"  D2 range: {df_sys['D2'].min():.4f} - {df_sys['D2'].max():.4f}")

df_all = pd.DataFrame(results_all)
df_all.to_csv(os.path.join(OUT_DIR, "boundary_alignment_multi.csv"), index=False)
print(f"\nSaved: {OUT_DIR}boundary_alignment_multi.csv")

print("\n" + "=" * 80)
print("FINAL ANALYSIS")
print("=" * 80)

df_log = df_all[df_all['system'] == 'LOGISTIC']
df_kur = df_all[df_all['system'] == 'KURAMOTO']
df_ou = df_all[df_all['system'] == 'OU']

print("\n1. LOGISTIC (chaos boundary at r ≈ 3.569):")
if df_log['collapse'].any():
    collapse_idx = np.where(df_log['collapse'].values)[0]
    print(f"   Collapse at r values: {df_log.iloc[collapse_idx]['param'].values}")
    
    lambda_arr = df_log['lambda'].values
    sign_changes = np.where(np.diff(np.sign(lambda_arr)))[0]
    if len(sign_changes) > 0:
        print(f"   λ sign changes: YES (at indices {sign_changes})")
    else:
        print(f"   λ sign changes: NO")

print("\n2. KURAMOTO (sync boundary at K ≈ 1.0):")
if df_kur['collapse'].any():
    collapse_idx = np.where(df_kur['collapse'].values)[0]
    print(f"   Collapse at K values: {df_kur.iloc[collapse_idx]['param'].values}")
else:
    print(f"   Collapse: NO")

print("\n3. OU (continuous - no boundary):")
if df_ou['collapse'].any():
    collapse_idx = np.where(df_ou['collapse'].values)[0]
    print(f"   Collapse at theta values: {df_ou.iloc[collapse_idx]['param'].values}")
else:
    print(f"   Collapse: NO")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

log_has_collapse = df_log['collapse'].any()
kur_has_collapse = df_kur['collapse'].any()
ou_has_collapse = df_ou['collapse'].any()

print(f"\nLogistic collapse: {'YES' if log_has_collapse else 'NO'}")
print(f"Kuramoto collapse: {'YES' if kur_has_collapse else 'NO'}")
print(f"OU collapse: {'YES' if ou_has_collapse else 'NO'}")

if log_has_collapse and not kur_has_collapse and not ou_has_collapse:
    print("\nD(k) collapse specifically detects chaos, NOT continuous transitions")
elif log_has_collapse:
    print("\nNeed to analyze alignment quality")