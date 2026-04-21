#!/usr/bin/env python3
"""
SFH-SGP_TASK_CLASSIFICATION
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

DATA_ROOT = "/home/student/sgp-tribe3/empirical_analysis/data/ds000030_s3/"
SUBJ_CSV = "/home/student/sgp-tribe3/empirical_analysis/ds000030_qc/clean_subjects.csv"
OUT_DIR = "/home/student/sgp-tribe3/empirical_analysis/outputs/"

TASKS = ['rest', 'bart', 'scap', 'stopsignal', 'taskswitch']
K_VALUES = [2, 4, 8, 16]
MAX_VOXELS = 3000
MAX_SUBJECTS = 10

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
print("STEP 1: Compute D(k) VECTORS")
print("=" * 70)

subj_list = pd.read_csv(SUBJ_CSV)['subject'].tolist()[:MAX_SUBJECTS]
print(f"Using {len(subj_list)} subjects")

results = []

for idx, subj in enumerate(subj_list):
    print(f"[{idx+1}/{len(subj_list)}] {subj}", end=" ", flush=True)
    
    subj_dk = {}
    
    for task in TASKS:
        path = f"{DATA_ROOT}/{subj}/func/{subj}_task-{task}_bold.nii.gz"
        
        img = nib.load(path)
        arr = np.asarray(img.get_fdata(), dtype=np.float32)
        n_tp = arr.shape[3]
        del img
        
        ts = arr.reshape(-1, n_tp).T
        del arr
        
        ts = ts[5:]
        
        var_vals = np.var(ts, axis=0)
        good_mask = var_vals > 0
        ts = ts[:, good_mask]
        
        if ts.shape[1] > MAX_VOXELS:
            sel = np.random.choice(ts.shape[1], MAX_VOXELS, replace=False)
            ts = ts[:, sel]
        
        for j in range(ts.shape[1]):
            m = np.mean(ts[:, j])
            s = np.std(ts[:, j])
            if s > 0:
                ts[:, j] = (ts[:, j] - m) / s
        
        dk_vals = {}
        for k in K_VALUES:
            dk_vals[k] = compute_knn_dim(ts, k)
        
        subj_dk[task] = dk_vals
        
        print(".", end="", flush=True)
        del ts
    
    for task in TASKS:
        results.append({
            'subject': subj,
            'task': task,
            'D2': subj_dk[task][2],
            'D4': subj_dk[task][4],
            'D8': subj_dk[task][8],
            'D16': subj_dk[task][16]
        })
    
    print(flush=True)

df = pd.DataFrame(results)
dk_cols = ['D2', 'D4', 'D8', 'D16']

print("\n" + "=" * 70)
print("STEP 2: BUILD DATASET - SAMPLE FEATURE VECTORS")
print("=" * 70)

print("\nSample D(k) vectors:")
for subj in subj_list[:3]:
    print(f"\n{subj}:")
    for task in TASKS:
        row = df[(df.subject == subj) & (df.task == task)].iloc[0]
        vec = [row['D2'], row['D4'], row['D8'], row['D16']]
        print(f"  {task:12s}: {vec}")

nan_count = df[dk_cols].isna().sum().sum()
print(f"\nNaN count: {nan_count}")
if nan_count > 0:
    print("FAIL: NaN detected")
    exit(1)

X = df[dk_cols].values
y = df['task'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n" + "=" * 70)
print("STEP 3: Classification (Logistic Regression, 5-fold CV)")
print("=" * 70)

chance_level = 1.0 / len(TASKS)
print(f"\nChance level: {chance_level:.2f} (1/{len(TASKS)})")

clf = LogisticRegression(max_iter=1000, random_state=42)

from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_pred = cross_val_predict(clf, X_scaled, y, cv=cv)
accuracy = accuracy_score(y, y_pred)

print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

print("\nConfusion Matrix:")
cm = confusion_matrix(y, y_pred)
print(f"\n{'Task':<12}", end="")
for t in TASKS:
    print(f" {t[:4]:>6}", end="")
print()
for i, t in enumerate(TASKS):
    print(f" {t:<12}", end="")
    for j in range(len(TASKS)):
        print(f" {cm[i,j]:>6}", end="")
    print()

print("\nPer-task accuracy:")
for i, t in enumerate(TASKS):
    mask = y == t
    task_acc = accuracy_score(y[mask], y_pred[mask])
    print(f"  {t:12s}: {task_acc:.4f}")

print("\n" + "=" * 70)
print("STEP 5: SHUFFLE CONTROL")
print("=" * 70)

np.random.seed(42)
y_shuffled = np.random.permutation(y)

y_pred_shuffled = cross_val_predict(clf, X_scaled, y_shuffled, cv=cv)
accuracy_shuffled = accuracy_score(y_shuffled, y_pred_shuffled)

print(f"\nShuffled accuracy: {accuracy_shuffled:.4f} ({accuracy_shuffled*100:.1f}%)")

print("\n" + "=" * 70)
print("STEP 6: INTERPRETATION")
print("=" * 70)

print(f"\n{'Metric':<30} {'REAL':<15} {'SHUFFLED':<15}")
print("-" * 60)
print(f"{'Accuracy':<30} {accuracy:<15.4f} {accuracy_shuffled:<15.4f}")
print(f"{'Chance level':<30} {chance_level:<15.4f}")

if accuracy <= chance_level + 0.05:
    print("\nRESULT: accuracy ≈ chance → metric NOT task-informative")
else:
    print(f"\nRESULT: accuracy = {accuracy:.2f} >> chance {chance_level:.2f}")
    if accuracy > accuracy_shuffled + 0.05:
        print("VALID signal: real >> shuffled")
    else:
        print("WEAK: real ≈ shuffled (may be artifact)")