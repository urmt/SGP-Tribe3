#!/usr/bin/env python3
"""
SFH-SGP_NEURAL_NETWORK_POPULATION_CLASSIFICATION
Test D(k) encoding of model identity using population sampling
"""
import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

N_SAMPLES = 200
N_REPEATS = 50
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

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("STEP 1: LOAD MNIST DATA")
print("=" * 70)

digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

print("\n" + "=" * 70)
print("STEP 2: TRAIN 4 MODELS")
print("=" * 70)

models = {}

print("\n1. LOGREG")
model1 = LogisticRegression(max_iter=1000, random_state=42)
model1.fit(X_train, y_train)
acc1 = model1.score(X_test, y_test)
print(f"  Accuracy: {acc1:.4f}")
if acc1 < 0.90:
    print("FAIL: LOGREG < 90%")
    exit(1)
models['LOGREG'] = model1

print("\n2. MLP_SMALL [32]")
model2 = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42, early_stopping=True)
model2.fit(X_train, y_train)
acc2 = model2.score(X_test, y_test)
print(f"  Accuracy: {acc2:.4f}")
if acc2 < 0.90:
    print("FAIL: MLP_SMALL < 90%")
    exit(1)
models['MLP_SMALL'] = model2

print("\n3. MLP_MEDIUM [64,32]")
model3 = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True)
model3.fit(X_train, y_train)
acc3 = model3.score(X_test, y_test)
print(f"  Accuracy: {acc3:.4f}")
if acc3 < 0.90:
    print("FAIL: MLP_MEDIUM < 90%")
    exit(1)
models['MLP_MEDIUM'] = model3

print("\n4. MLP_DEEP [128,64,32,16]")
model4 = MLPClassifier(hidden_layer_sizes=(128, 64, 32, 16), max_iter=500, random_state=42, early_stopping=True)
model4.fit(X_train, y_train)
acc4 = model4.score(X_test, y_test)
print(f"  Accuracy: {acc4:.4f}")
if acc4 < 0.90:
    print("FAIL: MLP_DEEP < 90%")
    exit(1)
models['MLP_DEEP'] = model4

print("\n" + "=" * 70)
print("STEP 3: EXTRACT FINAL HIDDEN LAYER")
print("=" * 70)

def get_final_hidden(model, X):
    if hasattr(model, 'coefs_'):
        acts = X.copy()
        for i, (w, b) in enumerate(zip(model.coefs_, model.intercepts_)):
            acts = np.dot(acts, w) + b
            if i < len(model.coefs_) - 1:
                acts = np.maximum(acts, 0)
        return acts
    return model.predict_proba(X)

for name, model in models.items():
    hidden = get_final_hidden(model, X_test[:10])
    print(f"{name}: {hidden.shape}")

print("\n" + "=" * 70)
print("STEP 4: POPULATION SAMPLING")
print("=" * 70)

all_samples = []

for model_name, model in models.items():
    print(f"\n{model_name}:")
    hidden_all = get_final_hidden(model, X_test)
    
    for rep in range(N_REPEATS):
        idx = np.random.choice(len(X_test), N_SAMPLES, replace=False)
        sample_hidden = hidden_all[idx]
        
        dk_vec = {}
        for k in K_VALUES:
            dk_vec[k] = compute_knn_dim(sample_hidden, k)
        
        all_samples.append({
            'model': model_name,
            'rep': rep,
            'D2': dk_vec[2],
            'D4': dk_vec[4],
            'D8': dk_vec[8],
            'D16': dk_vec[16]
        })
        
        if rep < 5:
            print(f"  rep{rep}: D2={dk_vec[2]:.3f}, D4={dk_vec[4]:.3f}, D8={dk_vec[8]:.3f}, D16={dk_vec[16]:.3f}")

df = pd.DataFrame(all_samples)
dk_cols = ['D2', 'D4', 'D8', 'D16']

print("\n" + "=" * 70)
print("STEP 5: SANITY CHECK")
print("=" * 70)

total_samples = len(df)
samples_per_model = (df['model'] == 'LOGREG').sum()

print(f"Total samples: {total_samples}")
print(f"Samples per model: {samples_per_model}")

if total_samples != 200:
    print(f"FAIL: Expected 200, got {total_samples}")
    exit(1)

nan_count = df[dk_cols].isna().sum().sum()
print(f"NaN count: {nan_count}")
if nan_count > 0:
    print("FAIL: NaN detected")
    exit(1)

std_vals = df[dk_cols].std()
print(f"D(k) std across all: {std_vals.to_dict()}")

if (std_vals == 0).any():
    print("FAIL: constant values detected")
    exit(1)

print("\nSample D(k) vectors (first 5):")
print(df.head()[dk_cols].to_string())

print("\n" + "=" * 70)
print("STEP 6: CLASSIFICATION")
print("=" * 70)

X = df[dk_cols].values
y = df['model'].values

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler_c = StandardScaler()
X_train_c = scaler_c.fit_transform(X_train_c)
X_test_c = scaler_c.transform(X_test_c)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_c, y_train_c)
y_pred = clf.predict(X_test_c)

real_acc = accuracy_score(y_test_c, y_pred)
print(f"REAL accuracy (single split): {real_acc:.4f}")

print("\nConfusion Matrix:")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_c, y_pred, labels=['LOGREG', 'MLP_SMALL', 'MLP_MEDIUM', 'MLP_DEEP'])
models_l = ['LOGREG', 'MLP_SMALL', 'MLP_MEDIUM', 'MLP_DEEP']
print(f"\n{'Model':<12}", end="")
for m in models_l:
    print(f" {m[:6]:>8}", end="")
print()
for i, m in enumerate(models_l):
    print(f" {m:<12}", end="")
    for j in range(4):
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

X_train_s = scaler_c.fit_transform(X_train_s)
X_test_s = scaler_c.transform(X_test_s)

clf_s = LogisticRegression(max_iter=1000, random_state=42)
clf_s.fit(X_train_s, y_train_s)
y_pred_s = clf_s.predict(X_test_s)

shuffled_acc = accuracy_score(y_test_s, y_pred_s)
print(f"SHUFFLED accuracy: {shuffled_acc:.4f}")

print("\n" + "=" * 70)
print("STEP 8: OUTPUT")
print("=" * 70)

chance = 0.25

print(f"\n{'Metric':<30} {'REAL':<15} {'SHUFFLED':<15}")
print("-" * 60)
print(f"{'Accuracy':<30} {real_acc:<15.4f} {shuffled_acc:<15.4f}")
print(f"{'Chance level':<30} {chance:<15.4f}")

print("\n" + "=" * 70)
print("STEP 9: STATISTICAL TEST (20 repeats)")
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
print(f"t-test: t = {t_stat:.4f}, p = {p_val:.4f}")

print("\n" + "=" * 70)
print("STEP 10: SAVE OUTPUTS")
print("=" * 70)

df.to_csv(os.path.join(OUT_DIR, "nn_population_classification.csv"), index=False)
print(f"Saved: {OUT_DIR}nn_population_classification.csv")

summary = f"real_mean={real_mean}\nreal_std={real_std}\nshuff_mean={shuff_mean}\nshuff_std={shuff_std}\nt={t_stat}\np={p_val}\n"
with open(os.path.join(OUT_DIR, "nn_population_summary.txt"), 'w') as f:
    f.write(summary)
print(f"Saved: {OUT_DIR}nn_population_summary.txt")

print("\n" + "=" * 70)
print("STEP 11: FINAL VALIDATION")
print("=" * 70)

varies = df[dk_cols].std().sum() > 0.01

print(f"\n1. Models trained fresh? YES")
print(f"2. Any fallback data? NO")
print(f"3. Total D(k) samples = 200? {'YES' if total_samples == 200 else 'NO'}")
print(f"4. D(k) varies across samples? {'YES' if varies else 'NO'}")
print(f"5. REAL > SHUFFLED? {'YES' if real_mean > shuff_mean else 'NO'}")

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)

if real_mean > shuff_mean + 0.05 and real_mean > chance + 0.05:
    print(f"PASS: D(k) encodes model identity")
    print(f"REAL ({real_mean:.2f}) >> SHUFFLED ({shuff_mean:.2f}) and > chance ({chance:.2f})")
else:
    print(f"INCONCLUSIVE: REAL ({real_mean:.2f}) vs SHUFFLED ({shuff_mean:.2f})")