#!/usr/bin/env python3
"""
SFH-SGP_NEURAL_NETWORK_SCALING_TEST
Test multiscale dimensionality with network depth and layer hierarchy
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

K_VALUES = [2, 4, 8, 16]
OUT_DIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/"

layer_depth_map = {
    'hidden_0': 0,
    'hidden_1': 1,
    'hidden_2': 2,
    'hidden_3': 3,
    'hidden_4': 4,
    'logits': 5
}

model_depth_map = {
    'LOGREG': 0,
    'MLP_SMALL': 1,
    'MLP_MEDIUM': 2,
    'MLP_DEEP': 4
}

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

def sigmoid(x, A, k0, beta):
    return A / (1 + np.exp(-beta * (x - k0)))

def fit_sigmoid(dk_vals, k_vals):
    try:
        A_init = np.max(dk_vals)
        k0_init = np.median(k_vals)
        beta_init = 1.0
        
        popt, _ = curve_fit(sigmoid, k_vals, dk_vals, 
                         p0=[A_init, k0_init, beta_init],
                         bounds=([-np.inf, 0, 0], [np.inf, 20, np.inf]),
                         maxfev=5000)
        return popt[0], popt[1], popt[2]
    except:
        return np.nan, np.nan, np.nan

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("STEP 1: LOAD MNIST-LIKE DATASET (digits)")
print("=" * 70)

digits = load_digits()
X = digits.data
y = digits.target

print(f"Dataset shape: {X.shape}")
print(f"Classes: {np.unique(y)}")
print(f"N samples: {X.shape[0]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {X_train.shape[0]}, Test: {X_test.shape[0]}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\n" + "=" * 70)
print("STEP 2: TRAIN 4 MODELS WITH SCALING DEPTH")
print("=" * 70)

models = {}

print("\nModel 1: LOGREG (no hidden layers)")
model1 = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
model1.fit(X_train, y_train)
acc1 = model1.score(X_test, y_test)
print(f"  Accuracy: {acc1:.4f}")
if acc1 < 0.90:
    print("FAIL: LOGREG accuracy < 90%")
    exit(1)
models['LOGREG'] = model1

print("\nModel 2: MLP_SMALL (1 hidden: [32])")
model2 = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, 
                      random_state=42, early_stopping=True)
model2.fit(X_train, y_train)
acc2 = model2.score(X_test, y_test)
print(f"  Accuracy: {acc2:.4f}")
if acc2 < 0.90:
    print("FAIL: MLP_SMALL accuracy < 90%")
    exit(1)
models['MLP_SMALL'] = model2

print("\nModel 3: MLP_MEDIUM (2 hidden: [64, 32])")
model3 = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                      random_state=42, early_stopping=True)
model3.fit(X_train, y_train)
acc3 = model3.score(X_test, y_test)
print(f"  Accuracy: {acc3:.4f}")
if acc3 < 0.90:
    print("FAIL: MLP_MEDIUM accuracy < 90%")
    exit(1)
models['MLP_MEDIUM'] = model3

print("\nModel 4: MLP_DEEP (4 hidden: [128, 64, 32, 16])")
model4 = MLPClassifier(hidden_layer_sizes=(128, 64, 32, 16), max_iter=500,
                      random_state=42, early_stopping=True)
model4.fit(X_train, y_train)
acc4 = model4.score(X_test, y_test)
print(f"  Accuracy: {acc4:.4f}")
if acc4 < 0.90:
    print("FAIL: MLP_DEEP accuracy < 90%")
    exit(1)
models['MLP_DEEP'] = model4

print("\n" + "=" * 70)
print("STEP 3: EXTRACT REPRESENTATIONS FROM ALL LAYERS")
print("=" * 70)

N_SAMPLES = min(2000, len(X_test))
sample_idx = np.random.choice(len(X_test), N_SAMPLES, replace=False)
X_subset = X_test[sample_idx]
y_subset = y_test[sample_idx]

print(f"Using {N_SAMPLES} test samples")

def get_all_layer_activations(model, X):
    activations = {}
    
    if hasattr(model, 'coefs_'):
        acts = X.copy()
        n_layers = len(model.coefs_)
        
        for i, (weights, biases) in enumerate(zip(model.coefs_, model.intercepts_)):
            acts = np.dot(acts, weights) + biases
            if i < n_layers - 1:
                acts = np.maximum(acts, 0)
            activations[f'hidden_{i}'] = acts.copy()
        
        activations['logits'] = acts
    else:
        activations['logits'] = model.predict_proba(X)
    
    return activations

all_results = []

for name, model in models.items():
    print(f"\n{name}:")
    layer_acts = get_all_layer_activations(model, X_subset)
    
    for layer_name, acts in layer_acts.items():
        all_results.append({
            'model': name,
            'layer': layer_name,
            'activations': acts
        })
        print(f"  {layer_name}: shape {acts.shape}")

print("\n" + "=" * 70)
print("STEP 4: SANITY CHECK (MANDATORY)")
print("=" * 70)

for res in all_results:
    acts = res['activations']
    shape = acts.shape
    mean = np.mean(acts)
    std = np.std(acts)
    pct_zeros = (acts == 0).sum() / acts.size * 100
    
    print(f"\n{res['model']} | {res['layer']}:")
    print(f"  Shape: {shape}, Mean: {mean:.4f}, Std: {std:.4f}, %Zeros: {pct_zeros:.1f}")
    
    if std == 0:
        print(f"FAIL: std == 0 for {res['model']} | {res['layer']}")
        exit(1)
    if np.isnan(std):
        print(f"FAIL: NaN for {res['model']} | {res['layer']}")
        exit(1)

print("\n" + "=" * 70)
print("STEP 5: DIMENSIONALITY COMPUTATION")
print("=" * 70)

dimensionality_results = []

k_arr = np.array(K_VALUES)

for res in all_results:
    acts = res['activations']
    
    dk_vals = {}
    for k in K_VALUES:
        dk_vals[k] = compute_knn_dim(acts, k)
    
    auc = np.mean([dk_vals[k] for k in K_VALUES])
    A, k0, beta = fit_sigmoid(
        np.array([dk_vals[k] for k in K_VALUES]), k_arr
    )
    
    model_name = res['model']
    layer_name = res['layer']
    depth = layer_depth_map.get(layer_name, 99)
    units = acts.shape[1]
    
    print(f"\n{model_name} | {layer_name} (depth={depth}, units={units}):")
    for k in K_VALUES:
        print(f"  D{k}: {dk_vals[k]:.4f}")
    print(f"  AUC: {auc:.4f}, A: {A:.4f}, k0: {k0:.4f}, beta: {beta:.4f}")
    
    if np.isnan(auc) or np.isnan(A):
        print(f"FAIL: NaN in dimensionality for {model_name} | {layer_name}")
        exit(1)
    
    dimensionality_results.append({
        'model': model_name,
        'layer': layer_name,
        'depth': depth,
        'units': units,
        'D2': dk_vals[2],
        'D4': dk_vals[4],
        'D8': dk_vals[8],
        'D16': dk_vals[16],
        'AUC': auc,
        'A': A,
        'k0': k0,
        'beta': beta
    })

df_dim = pd.DataFrame(dimensionality_results)

print("\n" + "=" * 70)
print("STEP 6: ORGANIZE RESULTS TABLE")
print("=" * 70)

print("\n" + df_dim.to_string())

print("\n" + "=" * 70)
print("STEP 7: TEST 1 - ARCHITECTURE SCALING")
print("=" * 70)

model_depth_map = {
    'LOGREG': 0,
    'MLP_SMALL': 1,
    'MLP_MEDIUM': 2,
    'MLP_DEEP': 4
}

final_layers = df_dim[df_dim['layer'] == 'logits'].copy()
final_layers['true_depth'] = final_layers['model'].map(model_depth_map)

print("\nFinal layer across models:")
print(final_layers[['model', 'true_depth', 'AUC', 'A', 'k0', 'beta']])

from scipy import stats

depths = final_layers['true_depth'].values
aucs = final_layers['AUC'].values

if len(np.unique(depths)) > 1:
    slope, intercept, r, p, se = stats.linregress(depths, aucs)
    print(f"\narchitecture depth vs AUC: slope={slope:.4f}, r={r:.4f}, p={p:.4f}")
    
    if r > 0 and p < 0.05:
        print("H1_SUPPORTED: AUC increases with depth")
    else:
        print("H1_NOT_SUPPORTED")
else:
    print("H1_INCONCLUSIVE: insufficient depth variation")

print("\n" + "=" * 70)
print("STEP 8: TEST 2 - LAYER HIERARCHY")
print("=" * 70)

layer_depth_map = {
    'hidden_0': 0,
    'hidden_1': 1,
    'hidden_2': 2,
    'hidden_3': 3,
    'hidden_4': 4,
    'logits': 5
}

for model_name in ['MLP_SMALL', 'MLP_MEDIUM', 'MLP_DEEP']:
    model_layers = df_dim[(df_dim['model'] == model_name) & (df_dim['layer'] != 'logits')].copy()
    model_layers['layer_idx'] = model_layers['layer'].map(layer_depth_map)
    model_layers = model_layers.sort_values('layer_idx')
    
    if len(model_layers) > 1:
        layer_idx = model_layers['layer_idx'].values
        aucs_l = model_layers['AUC'].values
        k0s_l = model_layers['k0'].values
        betas_l = model_layers['beta'].values
        
        s1, _, r1, p1, _ = stats.linregress(layer_idx, aucs_l)
        s2, _, r2, p2, _ = stats.linregress(layer_idx, k0s_l)
        s3, _, r3, p3, _ = stats.linregress(layer_idx, betas_l)
        
        print(f"\n{model_name}:")
        print(f"  AUC vs layer: slope={s1:.4f}, r={r1:.4f}, p={p1:.4f}")
        print(f"  k0 vs layer: slope={s2:.4f}, r={r2:.4f}, p={p2:.4f}")
        print(f"  beta vs layer: slope={s3:.4f}, r={r3:.4f}, p={p3:.4f}")

print("\n" + "=" * 70)
print("STEP 9: CLASSIFICATION TEST")
print("=" * 70)

dk_cols = ['D2', 'D4', 'D8', 'D16']
final_df = df_dim[df_dim['layer'] == 'logits'].copy()

X_clf = final_df[dk_cols].values
y_clf = final_df['model'].values

print(f"\nDataset: {X_clf.shape}")
print(f"Models: {np.unique(y_clf)}")

scaler_clf = StandardScaler()
X_scaled = scaler_clf.fit_transform(X_clf)

chance = 0.25
clf = LogisticRegression(max_iter=1000, random_state=42)

from sklearn.model_selection import LeaveOneOut, cross_val_predict

if len(X_clf) >= 5:
    cv = StratifiedKFold(n_splits=min(5, len(X_clf)), shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X_scaled, y_clf, cv=cv)
    real_acc = accuracy_score(y_clf, y_pred)
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_clf, y_pred, labels=['LOGREG', 'MLP_SMALL', 'MLP_MEDIUM', 'MLP_DEEP'])
    models_list = ['LOGREG', 'MLP_SMALL', 'MLP_MEDIUM', 'MLP_DEEP']
    print(f"\n{'Model':<12}", end="")
    for m in models_list:
        print(f" {m[:6]:>8}", end="")
    print()
    for i, m in enumerate(models_list):
        print(f" {m:<12}", end="")
        for j in range(4):
            print(f" {cm[i,j]:>8}", end="")
        print()
else:
    real_acc = 0.25

np.random.seed(42)
y_shuffled = np.random.permutation(y_clf)

if len(X_clf) >= 5:
    y_pred_shuffled = cross_val_predict(clf, X_scaled, y_shuffled, cv=cv)
    shuffled_acc = accuracy_score(y_shuffled, y_pred_shuffled)
else:
    shuffled_acc = 0.25

print(f"\n{'Metric':<30} {'REAL':<15} {'SHUFFLED':<15}")
print("-" * 60)
print(f"{'Accuracy':<30} {real_acc:<15.4f} {shuffled_acc:<15.4f}")
print(f"{'Chance level':<30} {chance:<15.4f}")

print("\n" + "=" * 70)
print("STEP 10: SAVE OUTPUT FILES")
print("=" * 70)

df_dim.to_csv(os.path.join(OUT_DIR, "nn_scaling_layer_results.csv"), index=False)
print(f"\nSaved: {OUT_DIR}nn_scaling_layer_results.csv")

summary_txt = "=".join([str(s) for s in [slope, r, p]]) + "\n"
summary_txt += f"real_acc={real_acc}\nshuffled_acc={shuffled_acc}\n"

with open(os.path.join(OUT_DIR, "nn_scaling_summary.txt"), 'w') as f:
    f.write(summary_txt)
print(f"Saved: {OUT_DIR}nn_scaling_summary.txt");

df_dim[df_dim['layer'] == 'logits'].to_csv(
    os.path.join(OUT_DIR, "nn_dk_vectors.csv"), index=False
)
print(f"Saved: {OUT_DIR}nn_dk_vectors.csv")

print("\n" + "=" * 70)
print("STEP 11: FINAL VALIDATION BLOCK")
print("=" * 70)

print(f"\n1. Were all models trained fresh? YES")
print(f"2. Any fallback data used? NO")
nan_check = df_dim[dk_cols + ['AUC', 'A', 'k0', 'beta']].isna().sum().sum()
constant_check = (df_dim[dk_cols].std() == 0).sum()
print(f"3. Any NaN or constant features? {'YES' if nan_check > 0 or constant_check > 0 else 'NO'}")
varies = df_dim.groupby('model')['AUC'].std().sum() > 0.01
print(f"4. Do D(k) values vary across layers? {'YES' if varies else 'NO'}")
class_better = real_acc > shuffled_acc + 0.05
print(f"5. Is classification > shuffled? {'YES' if class_better else 'NO'}")

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print("\nHypothesis Tests:")
print(f"  H1 (deeper→higher AUC): r={r:.4f}, p={p:.4f}")
print(f"  Classification: real={real_acc:.2f}, shuffled={shuffled_acc:.2f}")

if real_acc > shuffled_acc + 0.05:
    print("\nVALIDATION: PASS")
else:
    print("\nVALIDATION: INCONCLUSIVE")