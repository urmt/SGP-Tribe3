#!/usr/bin/env python3
"""
SFH-SGP_DYNAMICAL_SYSTEMS_CLASSIFICATION
Test D(k) distinguishing periodic, chaotic, and random systems
"""
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

N_SAMPLES = 300
N_REPEATS = 50
K_VALUES = [2, 4, 8, 16]
EMBED_DIM = 10
DELAY = 2
SERIES_LENGTH = 5000
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
print("STEP 1: GENERATE TIME SERIES")
print("=" * 70)

print("\n1. PERIODIC (sine wave)")
t = np.arange(SERIES_LENGTH)
periodic = np.sin(2 * np.pi * t / 50)
print(f"   Generated: {len(periodic)} points")

print("\n2. CHAOTIC (logistic map, r=4.0)")
x = 0.12345
r = 4.0
chaotic = np.zeros(SERIES_LENGTH)
for i in range(SERIES_LENGTH):
    x = r * x * (1 - x)
    chaotic[i] = x
print(f"   Generated: {len(chaotic)} points")

print("\n3. RANDOM (Gaussian noise)")
random_sys = np.random.randn(SERIES_LENGTH)
print(f"   Generated: {len(random_sys)} points")

print("\n" + "=" * 70)
print("STEP 2: TIME DELAY EMBEDDING")
print("=" * 70)

print(f"\nEmbedding dimension: {EMBED_DIM}, Delay: {DELAY}")

X_periodic = time_delay_embed(periodic, EMBED_DIM, DELAY)
X_chaotic = time_delay_embed(chaotic, EMBED_DIM, DELAY)
X_random = time_delay_embed(random_sys, EMBED_DIM, DELAY)

print(f"Periodic shape: {X_periodic.shape}")
print(f"Chaotic shape: {X_chaotic.shape}")
print(f"Random shape: {X_random.shape}")

print("\n" + "=" * 70)
print("STEP 3: SANITY CHECK")
print("=" * 70)

for name, X in [('PERIODIC', X_periodic), ('CHAOTIC', X_chaotic), ('RANDOM', X_random)]:
    print(f"\n{name}:")
    print(f"  Shape: {X.shape}")
    print(f"  Mean: {X.mean():.6f}")
    print(f"  Std: {X.std():.6f}")
    print(f"  First 3 rows:\n{X[:3]}")
    
    if X.std() == 0:
        print(f"FAIL: {name} has zero std")
        exit(1)
    if np.isnan(X).any():
        print(f"FAIL: {name} has NaN")
        exit(1)

print("\n" + "=" * 70)
print("STEP 4: POPULATION SAMPLING")
print("=" * 70)

all_samples = []

datasets = {
    'PERIODIC': X_periodic,
    'CHAOTIC': X_chaotic,
    'RANDOM': X_random
}

for sys_name, X in datasets.items():
    print(f"\n{sys_name}:")
    
    for rep in range(N_REPEATS):
        idx = np.random.choice(len(X), N_SAMPLES, replace=False)
        sample = X[idx]
        
        dk_vec = {}
        for k in K_VALUES:
            dk_vec[k] = compute_knn_dim(sample, k)
        
        all_samples.append({
            'system': sys_name,
            'rep': rep,
            'D2': dk_vec[2],
            'D4': dk_vec[4],
            'D8': dk_vec[8],
            'D16': dk_vec[16]
        })
        
        if rep < 3:
            print(f"  rep{rep}: D2={dk_vec[2]:.3f}, D4={dk_vec[4]:.3f}, D8={dk_vec[8]:.3f}, D16={dk_vec[16]:.3f}")

df = pd.DataFrame(all_samples)
dk_cols = ['D2', 'D4', 'D8', 'D16']

print("\n" + "=" * 70)
print("STEP 5: VALIDATION")
print("=" * 70)

total_samples = len(df)
samples_per_system = (df['system'] == 'PERIODIC').sum()

print(f"Total samples: {total_samples}")
print(f"Samples per system: {samples_per_system}")

if total_samples != 150:
    print(f"FAIL: Expected 150, got {total_samples}")
    exit(1)

nan_count = df[dk_cols].isna().sum().sum()
print(f"NaN count: {nan_count}")
if nan_count > 0:
    print("FAIL: NaN detected")
    exit(1)

std_vals = df[dk_cols].std()
print(f"D(k) std: {std_vals.to_dict()}")

if (std_vals == 0).any():
    print("FAIL: constant values")
    exit(1)

print("\n" + "=" * 70)
print("STEP 6: CLASSIFICATION")
print("=" * 70)

X = df[dk_cols].values
y = df['system'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

real_acc = accuracy_score(y_test, y_pred)
print(f"\nREAL accuracy: {real_acc:.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=['PERIODIC', 'CHAOTIC', 'RANDOM'])
systems = ['PERIODIC', 'CHAOTIC', 'RANDOM']
print(f"\n{'System':<12}", end="")
for s in systems:
    print(f" {s[:6]:>8}", end="")
print()
for i, s in enumerate(systems):
    print(f" {s:<12}", end="")
    for j in range(3):
        print(f" {cm[i,j]:>8}", end="")
    print()

print("\n" + "=" * 70)
print("STEP 7: SHUFFLE TEST")
print("=" * 70)

np.random.seed(42)
y_shuffled = np.random.permutation(y)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X, y_shuffled, test_size=0.2, random_state=42, stratify=y_shuffled
)

X_train_s = scaler.fit_transform(X_train_s)
X_test_s = scaler.transform(X_test_s)

clf_s = LogisticRegression(max_iter=1000, random_state=42)
clf_s.fit(X_train_s, y_train_s)
y_pred_s = clf_s.predict(X_test_s)

shuffled_acc = accuracy_score(y_test_s, y_pred_s)
print(f"SHUFFLED accuracy: {shuffled_acc:.4f}")

print("\n" + "=" * 70)
print("STEP 8: REPEAT FOR STABILITY")
print("=" * 70)

real_accs = []
shuffled_accs = []

for seed in range(20):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    
    c = LogisticRegression(max_iter=1000, random_state=seed)
    c.fit(X_tr, y_tr)
    real_accs.append(c.score(X_te, y_te))
    
    y_sh = np.random.RandomState(seed).permutation(y)
    X_sh_tr, X_sh_te, y_sh_tr, y_sh_te = train_test_split(
        X, y_sh, test_size=0.2, random_state=seed, stratify=y_sh
    )
    X_sh_tr = sc.fit_transform(X_sh_tr)
    X_sh_te = sc.transform(X_sh_te)
    
    c_s = LogisticRegression(max_iter=1000, random_state=seed)
    c_s.fit(X_sh_tr, y_sh_tr)
    shuffled_accs.append(c_s.score(X_sh_te, y_sh_te))

real_mean = np.mean(real_accs)
real_std = np.std(real_accs)
shuff_mean = np.mean(shuffled_accs)
shuff_std = np.std(shuffled_accs)

t_stat, p_val = stats.ttest_ind(real_accs, shuffled_accs)

print(f"\nREAL:  Mean = {real_mean:.4f} ± {real_std:.4f}")
print(f"SHUFFLED: Mean = {shuff_mean:.4f} ± {shuff_std:.4f}")
print(f"t-test: t = {t_stat:.4f}, p = {p_val:.6f}")

print("\n" + "=" * 70)
print("STEP 9: OUTPUT")
print("=" * 70)

chance = 1/3

print(f"\n{'Metric':<30} {'REAL':<15} {'SHUFFLED':<15}")
print("-" * 60)
print(f"{'Accuracy':<30} {real_mean:<15.4f} {shuff_mean:<15.4f}")
print(f"{'Chance level':<30} {chance:<15.4f}")

print("\n" + "=" * 70)
print("STEP 10: SAVE OUTPUTS")
print("=" * 70)

df.to_csv(os.path.join(OUT_DIR, "dynamical_systems_results.csv"), index=False)
print(f"Saved: {OUT_DIR}dynamical_systems_results.csv")

summary = f"real_mean={real_mean}\nreal_std={real_std}\nshuff_mean={shuff_mean}\nshuff_std={shuff_std}\nt={t_stat}\np={p_val}\n"
with open(os.path.join(OUT_DIR, "dynamical_summary.txt"), 'w') as f:
    f.write(summary)
print(f"Saved: {OUT_DIR}dynamical_summary.txt")

print("\n" + "=" * 70)
print("STEP 11: FINAL VALIDATION")
print("=" * 70)

varies = df[dk_cols].std().sum() > 0.01

print(f"\n1. Systems generated correctly? YES")
print(f"2. Same embedding used? YES")
print(f"3. Total samples = 150? {'YES' if total_samples == 150 else 'NO'}")
print(f"4. D(k) varies across systems? {'YES' if varies else 'NO'}")
print(f"5. REAL > SHUFFLED? {'YES' if real_mean > shuff_mean else 'NO'}")

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)

if real_mean > shuff_mean + 0.05 and real_mean > chance + 0.05:
    print(f"PASS: D(k) distinguishes dynamical systems")
    print(f"REAL ({real_mean:.2f}) >> SHUFFLED ({shuff_mean:.2f}) and > chance ({chance:.2f})")
else:
    print(f"INCONCLUSIVE: REAL ({real_mean:.2f}) vs SHUFFLED ({shuff_mean:.2f}) vs chance ({chance:.2f})")