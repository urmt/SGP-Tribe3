#!/usr/bin/env python3
"""
MULTITASK LOAD GRADIENT - STRICT VALIDATION
"""
import os, gc, numpy as np, pandas as pd, nibabel as nib
from scipy import stats
from sklearn.decomposition import PCA

DATA_ROOT = "/home/student/sgp-tribe3/empirical_analysis/data/ds000030_s3/"
SUBJ_CSV = "/home/student/sgp-tribe3/empirical_analysis/ds000030_qc/clean_subjects.csv"
OUT = "/home/student/sgp-tribe3/empirical_analysis/multitask_results/"

TASKS = ['rest', 'bart', 'scap', 'stopsignal', 'taskswitch']

print("=" * 70)
print("STEP 0: SUBJECT/TASK VALIDATION")
print("=" * 70)

subjs = pd.read_csv(SUBJ_CSV)['subject'].tolist()
valid_subjects = []
missing_tasks = []

for s in subjs:
    task_status = {}
    all_present = True
    
    for task in TASKS:
        path = f"{DATA_ROOT}/{s}/func/{s}_task-{task}_bold.nii.gz"
        present = os.path.exists(path)
        task_status[task] = present
        if not present:
            all_present = False
    
    if all_present:
        valid_subjects.append(s)
    else:
        missing = [t for t, p in task_status.items() if not p]
        missing_tasks.append((s, missing))

print(f"Total subjects in clean list: {len(subjs)}")
print(f"Subjects with ALL 5 tasks: {len(valid_subjects)}")
print(f"Subjects missing tasks: {len(missing_tasks)}")

if missing_tasks:
    print("\nMissing tasks:")
    for s, missing in missing_tasks[:5]:
        print(f"  {s}: {missing}")

if len(valid_subjects) < 40:
    print(f"\nWARNING: Only {len(valid_subjects)} valid subjects")

print("\n" + "=" * 70)
print("STEP 1: MULTITASK DIMENSIONALITY")
print("=" * 70)

def compute_dim(data, k):
    try:
        if data.shape[0] < k + 2:
            return np.nan
        p = PCA(min(k + 1, data.shape[0] - 1))
        p.fit(data)
        return np.sum(p.explained_variance_ratio_[:k])
    except:
        return np.nan

results = []

for si, s in enumerate(valid_subjects):
    print(f"[{si+1}/{len(valid_subjects)}] {s}", end=" ", flush=True)
    task_aucs = {}
    
    for task in TASKS:
        path = f"{DATA_ROOT}/{s}/func/{s}_task-{task}_bold.nii.gz"
        
        img = nib.load(path)
        arr = img.get_fdata()
        n_tp = arr.shape[3]
        del img
        
        ts = arr.reshape(-1, n_tp).T.astype(np.float32)
        del arr
        gc.collect()
        
        ts = ts[:, ::max(1, ts.shape[1] // 1500)]
        
        for j in range(ts.shape[1]):
            y = ts[:, j]
            ts[:, j] = (y - np.mean(y)) / (np.std(y) + 1e-10)
        
        d2 = compute_dim(ts, 2)
        d4 = compute_dim(ts, 4)
        auc = np.nanmean([d2, d4])
        
        task_aucs[task] = auc
        
        del ts
        gc.collect()
    
    results.append({'subject': s, **task_aucs})
    print("done", flush=True)

print(f"\nProcessed: {len(results)} subjects")

df = pd.DataFrame(results)

# STEP 2: ANALYSIS
print("\n" + "=" * 70)
print("STEP 2: ANALYSIS")
print("=" * 70)

task_means = {t: df[t].mean() for t in TASKS}
task_stds = {t: df[t].std() for t in TASKS}

print(f"\n{'Task':<12} {'Mean AUC':>10} {'Std':>10}")
print(f"{'-'*34}")
for t in TASKS:
    print(f"{t:<12} {task_means[t]:>10.4f} {task_stds[t]:>10.4f}")

# Load mapping
load_map = {'rest': 1, 'scap': 2, 'bart': 3, 'stopsignal': 4, 'taskswitch': 5}

auc_flat = []
load_flat = []
for _, row in df.iterrows():
    for t in TASKS:
        auc_flat.append(row[t])
        load_flat.append(load_map[t])

slope, intercept, r, p, se = stats.linregress(load_flat, auc_flat)

print(f"\nLinear Regression: AUC ~ Load")
print(f"  Slope: {slope:.5f}")
print(f"  r: {r:.4f}")
print(f"  p-value: {p:.4f}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if p < 0.05 and slope > 0:
    result = "YES"
elif p < 0.05:
    result = "DECREASES"
else:
    result = "NO"

print(f"Dimensionality increases with load: {result}")