#!/usr/bin/env python3
"""
SFH-SGP_NEURAL_NETWORK_DIMENSIONALITY_TEST
Test whether D(k) encodes MODEL IDENTITY in clean neural network system
"""
import os
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

K_VALUES = [2, 4, 8, 16]
OUT_DIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/"

np.random.seed(42)

def compute_knn_dim(data, k):
    try:
        n, d = data.shape
        if n <= k + 1 or d < 10:
            return np.nan
        nn = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree')
        nn.fit(data)
        dists, _ = nn.kneighbors(data)
        return np.mean(np.log(dists[:, -1] + 1e-10))
    except:
        return np.nan

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("STEP 1: LOAD DIGITS DATASET")
print("=" * 70)

digits = load_digits()
X = digits.data
y = digits.target

print(f"Dataset shape: {X.shape}")
print(f"Classes: {np.unique(y)}")
print(f"N samples: {X.shape[0]}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nTrain: {X_train.shape[0]}, Test: {X_test.shape[0]}")

print("\n" + "=" * 70)
print("STEP 2: TRAIN 3 DIFFERENT MODELS")
print("=" * 70)

models = {}

print("\nModel 1: Logistic Regression")
model1 = LogisticRegression(max_iter=1000, random_state=42)
model1.fit(X_train, y_train)
acc1 = model1.score(X_test, y_test)
print(f"  Accuracy: {acc1:.4f}")
models['LOGREG'] = model1

print("\nModel 2: MLP (1 hidden layer)")
model2 = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42)
model2.fit(X_train, y_train)
acc2 = model2.score(X_test, y_test)
print(f"  Accuracy: {acc2:.4f}")
models['MLP1'] = model2

print("\nModel 3: MLP (2 hidden layers)")
model3 = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
model3.fit(X_train, y_train)
acc3 = model3.score(X_test, y_test)
print(f"  Accuracy: {acc3:.4f}")
models['MLP2'] = model3

print("\n" + "=" * 70)
print("STEP 3: EXTRACT ACTIVATIONS")
print("=" * 70)

def get_activations(model, X_data):
    if hasattr(model, 'coefs_'):
        activations = X_data
        for i, (weights, biases) in enumerate(zip(model.coefs_, model.intercepts_)):
            activations = np.dot(activations, weights) + biases
            activations = np.maximum(activations, 0)
        return activations
    return X_data

print("\nExtracting hidden layer activations for all samples...")
model_activations = {}
for name, model in models.items():
    if name == 'LOGREG':
        model_activations[name] = model.predict_proba(X)
    else:
        model_activations[name] = get_activations(model, X)
    print(f"  {name}: {model_activations[name].shape}")

print("\n" + "=" * 70)
print("STEP 4: COMPUTE D(k)")
print("=" * 70)

print("\nSample activations:")
for name, acts in model_activations.items():
    print(f"\n{name}:")
    print(f"  Shape: {acts.shape}")
    print(f"  Mean: {acts.mean():.4f}, Std: {acts.std():.4f}")

print("\nComputing D(k) for each model...")

dk_results = []
for name, acts in model_activations.items():
    dk_vec = {}
    for k in K_VALUES:
        dk_vec[k] = compute_knn_dim(acts, k)
    print(f"\n{name} D(k):")
    for k in K_VALUES:
        print(f"  D{k}: {dk_vec[k]:.4f}")
    
    dk_results.append({
        'model': name,
        'D2': dk_vec[2],
        'D4': dk_vec[4],
        'D8': dk_vec[8],
        'D16': dk_vec[16]
    })

import pandas as pd

results_df = pd.DataFrame(dk_results)
print("\n\nDataset:")
print(results_df)

dk_cols = ['D2', 'D4', 'D8', 'D16']

print("\n" + "=" * 70)
print("STEP 5: BUILD CLASSIFICATION DATASET - PER-CLASS D(k)")
print("=" * 70)

print("\nComputing per-class D(k) for better statistics...")

per_class_data = []
for name, acts in model_activations.items():
    unique_classes = np.unique(y)
    
    for cls in unique_classes:
        mask = y == cls
        class_acts = acts[mask]
        
        if class_acts.shape[0] > 16:
            dk_vec = {}
            for k in K_VALUES:
                dk_vec[k] = compute_knn_dim(class_acts, k)
            
            per_class_data.append({
                'model': name,
                'digit': cls,
                'D2': dk_vec[2],
                'D4': dk_vec[4],
                'D8': dk_vec[8],
                'D16': dk_vec[16]
            })

per_df = pd.DataFrame(per_class_data)
print(f"\nTotal class-model pairs: {len(per_df)}")

print("\nSample D(k) vectors:")
for name in ['LOGREG', 'MLP1', 'MLP2']:
    sample = per_df[per_df.model == name].head(1)
    print(f"\n{name} (digit {sample['digit'].values[0]}):")
    print(f"  D2: {sample['D2'].values[0]:.4f}")
    print(f"  D4: {sample['D4'].values[0]:.4f}")
    print(f"  D8: {sample['D8'].values[0]:.4f}")
    print(f"  D16: {sample['D16'].values[0]:.4f}")

nan_count = per_df[dk_cols].isna().sum().sum()
print(f"\nNaN count: {nan_count}")
if nan_count > 0:
    print("FAIL: NaN detected")
    exit(1)

print("\n" + "=" * 70)
print("STEP 6: CLASSIFICATION TEST")
print("=" * 70)

X_dim = per_df[dk_cols].values
y_model = per_df['model'].values

print(f"\nX shape: {X_dim.shape}")
print(f"Labels distribution:")
for m in ['LOGREG', 'MLP1', 'MLP2']:
    print(f"  {m}: {(y_model == m).sum()}")

chance_level = 1.0 / len(np.unique(y_model))
print(f"Chance level: {chance_level:.4f}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_dim)

clf = LogisticRegression(max_iter=1000, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_pred_cv = cross_val_predict(clf, X_scaled, y_model, cv=cv)
real_acc = accuracy_score(y_model, y_pred_cv)

print(f"\nReal accuracy (5-fold CV): {real_acc:.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_model, y_pred_cv, labels=['LOGREG', 'MLP1', 'MLP2'])
print(f"\n{'Model':<10}", end="")
for m in ['LOGREG', 'MLP1', 'MLP2']:
    print(f" {m:<8}", end="")
print()
for i, m in enumerate(['LOGREG', 'MLP1', 'MLP2']):
    print(f" {m:<10}", end="")
    for j in range(3):
        print(f" {cm[i,j]:<8}", end="")
    print()

print("\n" + "=" * 70)
print("STEP 7: SHUFFLE CONTROL")
print("=" * 70)

np.random.seed(42)
y_shuffled = np.random.permutation(y_model)

y_pred_shuffled = cross_val_predict(clf, X_scaled, y_shuffled, cv=cv)
shuffled_acc = accuracy_score(y_shuffled, y_pred_shuffled)

print(f"\nShuffled accuracy: {shuffled_acc:.4f}")

print("\n" + "=" * 70)
print("STEP 8: OUTPUT")
print("=" * 70)

print(f"\n{'Metric':<30} {'REAL':<15} {'SHUFFLED':<15}")
print("-" * 60)
print(f"{'Accuracy':<30} {real_acc:<15.4f} {shuffled_acc:<15.4f}")
print(f"{'Chance level':<30} {chance_level:<15.4f}")

print("\n" + "=" * 70)
print("STEP 9: INTERPRETATION")
print("=" * 70)

if real_acc > shuffled_acc + 0.05 and real_acc > chance_level + 0.05:
    print(f"\nRESULT: real ({real_acc:.2f}) >> shuffled ({shuffled_acc:.2f}) and > chance ({chance_level:.2f})")
    print("PASS: D(k) encodes system architecture")
elif real_acc <= chance_level + 0.05:
    print(f"\nRESULT: real ≈ chance")
    print("FAIL: D(k) not sensitive")
else:
    print(f"\nRESULT: real ({real_acc:.2f}) vs shuffled ({shuffled_acc:.2f})")
    print("INCONCLUSIVE")

per_df.to_csv(os.path.join(OUT_DIR, "nn_dimension_results.csv"), index=False)
print(f"\nSaved: {OUT_DIR}nn_dimension_results.csv")