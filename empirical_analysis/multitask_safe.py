#!/usr/bin/env python3
"""
SFH-SGP_MULTITASK_REGIME_MAPPING_02_SAFE
Strict pipeline - one subject at a time
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats
from sklearn.neighbors import NearestNeighbors

DATA_ROOT = "/home/student/sgp-tribe3/empirical_analysis/data/ds000030_s3/"
SUBJ_CSV = "/home/student/sgp-tribe3/empirical_analysis/ds000030_qc/clean_subjects.csv"
OUT_DIR = "/home/student/sgp-tribe3/empirical_analysis/multitask_results/"

TASKS = ['rest', 'bart', 'scap', 'stopsignal', 'taskswitch']
MAX_VOXELS = 3000

def compute_knn_dim(data, k):
    try:
        n, d = data.shape
        if n <= k + 1 or d < 10:
            return np.nan
        nn = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree')
        nn.fit(data)
        dist, _ = nn.kneighbors(data)
        log_dist = np.log(dist[:, -1] + 1e-10)
        return np.mean(log_dist)
    except:
        return np.nan

os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(OUT_DIR, "multitask_results.csv")

print("=" * 70)
print("STEP 1: VALIDATION")
print("=" * 70)

subj_list = pd.read_csv(SUBJ_CSV)['subject'].tolist()
print(f"Total in CSV: {len(subj_list)}")

valid_subjects = []
for subj in subj_list:
    all_present = True
    sizes = []
    for task in TASKS:
        path = f"{DATA_ROOT}/{subj}/func/{subj}_task-{task}_bold.nii.gz"
        present = os.path.exists(path)
        sizes.append(os.path.getsize(path) / (1024*1024) if present else 0)
        if not present:
            all_present = False
    
    if all_present:
        valid_subjects.append(subj)
        print(f"{subj}: all 5 tasks")

print(f"\nValid: {len(valid_subjects)}")
if len(valid_subjects) < 40:
    print("FAIL")
    exit(1)

print("\n" + "=" * 70)
print("STEP 2-4: PROCESS")
print("=" * 70)

for idx, subj in enumerate(valid_subjects):
    print(f"[{idx+1}/{len(valid_subjects)}] {subj}")
    
    for task in TASKS:
        img = nib.load(f"{DATA_ROOT}/{subj}/func/{subj}_task-{task}_bold.nii.gz")
        arr = np.asarray(img.get_fdata(), dtype=np.float32)
        n_tp = arr.shape[3]
        del img
        
        ts = arr.reshape(-1, n_tp).T
        del arr
        ts = ts[5:]
        
        var_vals = np.var(ts, axis=0)
        good_mask = var_vals > 0
        ts = ts[:, good_mask]
        n_good = ts.shape[1]
        
        if n_good > MAX_VOXELS:
            sel_idx = np.random.choice(n_good, MAX_VOXELS, replace=False)
            ts = ts[:, sel_idx]
        
        for j in range(ts.shape[1]):
            m = np.mean(ts[:, j])
            s = np.std(ts[:, j])
            if s > 0:
                ts[:, j] = (ts[:, j] - m) / s
            else:
                ts[:, j] = 0
        
        d2 = compute_knn_dim(ts, 2)
        d4 = compute_knn_dim(ts, 4)
        
        if np.isnan(d2) or np.isnan(d4):
            print(f"FAIL NaN")
            exit(1)
        
        auc = (d2 + d4) / 2
        
        with open(RESULTS_FILE, 'a') as f:
            f.write(f"{subj},{task},{auc}\n")
        
        print(f"  {task}: auc={auc:.3f}")

print("\n" + "=" * 70)
print("STEP 5-7: ANALYSIS")
print("=" * 70)

df = pd.read_csv(RESULTS_FILE)
means = df.groupby('task')['AUC'].mean()
stds = df.groupby('task')['AUC'].std()

print("\nTask | Mean | Std")
for task in TASKS:
    print(f"{task:<12} {means[task]:.4f} {stds[task]:.4f}")

f_stat, p_val = stats.f_oneway(*[df[df.task==t]['AUC'] for t in TASKS])
print(f"\nANOVA: F={f_stat:.2f}, p={p_val:.4f}")

print("\nPairwise vs rest:")
rest_auc = df[df.task=='rest']['AUC']
for task in ['bart','scap','stopsignal','taskswitch']:
    t_auc = df[df.task==task]['AUC']
    t_stat, p_t = stats.ttest_rel(t_auc, rest_auc)
    d = (t_auc.mean() - rest_auc.mean()) / np.std(t_auc - rest_auc)
    print(f"  rest vs {task}: t={t_stat:.2f}, p={p_t:.4f}, d={d:.2f}")

pivot = df.pivot(index='subject', columns='task', values='AUC')
corr = pivot.T.corr()
print(f"\nSimilarity:\n{corr.round(2).to_string()}")

print(f"\nDimensionality differs: {'YES' if p_val < 0.05 else 'NO'}")