#!/usr/bin/env python3
"""
SFH-SGP_REGIME_STABILITY_AND_SEPARABILITY_01
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

DATA_ROOT = "/home/student/sgp-tribe3/empirical_analysis/data/ds000030_s3/"
SUBJ_CSV = "/home/student/sgp-tribe3/empirical_analysis/ds000030_qc/clean_subjects.csv"
OUT_DIR = "/home/student/sgp-tribe3/empirical_analysis/outputs/"
FIG_DIR = "/home/student/sgp-tribe3/empirical_analysis/figures/"

TASKS = ['rest', 'bart', 'scap', 'stopsignal', 'taskswitch']
MAX_VOXELS = 5000

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
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 70)
print("STEP 1: VALIDATION")
print("=" * 70)

subj_list = pd.read_csv(SUBJ_CSV)['subject'].tolist()

valid_subjects = []
for subj in subj_list:
    all_present = True
    sizes = []
    shapes = []
    for task in TASKS:
        path = f"{DATA_ROOT}/{subj}/func/{subj}_task-{task}_bold.nii.gz"
        if os.path.exists(path):
            sizes.append(os.path.getsize(path) / (1024*1024))
            img = nib.load(path)
            shapes.append(img.shape)
            del img
        else:
            all_present = False
            sizes.append(0)
            shapes.append(None)
    
    if all_present:
        valid_subjects.append(subj)

print(f"\nValidating first 5 subjects:")
for subj in valid_subjects[:5]:
    print(f"\n{subj}:")
    for i, task in enumerate(TASKS):
        path = f"{DATA_ROOT}/{subj}/func/{subj}_task-{task}_bold.nii.gz"
        print(f"  {task}: {path}")
        print(f"    size: {os.path.getsize(path)/(1024*1024):.1f}MB, shape: {shapes[i]}")

print(f"\nTotal valid subjects: {len(valid_subjects)}")

if len(valid_subjects) < 40:
    print("FAIL: need 40+ subjects")
    exit(1)

print("\n" + "=" * 70)
print("STEP 2-4: PREPROCESSING & DIMENSIONALITY")
print("=" * 70)

results = []

for idx, subj in enumerate(valid_subjects):
    print(f"[{idx+1}/{len(valid_subjects)}] {subj}", end=" ", flush=True)
    
    subj_results = {}
    
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
        
        d2 = compute_knn_dim(ts, 2)
        d4 = compute_knn_dim(ts, 4)
        d8 = compute_knn_dim(ts, 8)
        d16 = compute_knn_dim(ts, 16)
        
        early_auc = np.nanmean([d2, d4, d8])
        
        subj_results[task] = {
            'D2': d2, 'D4': d4, 'D8': d8, 'D16': d16,
            'early_auc': early_auc
        }
        
        print(".", end="", flush=True)
        
        del ts
    
    for task in TASKS:
        results.append({
            'subject': subj,
            'task': task,
            'D2': subj_results[task]['D2'],
            'D4': subj_results[task]['D4'],
            'D8': subj_results[task]['D8'],
            'D16': subj_results[task]['D16'],
            'early_auc': subj_results[task]['early_auc']
        })
    
    print(flush=True)

df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("STEP 5: WITHIN-TASK CONSISTENCY")
print("=" * 70)

task_correlations = []
for task in TASKS:
    task_df = df[df.task == task]
    dk_vecs = task_df[['D2', 'D4', 'D8', 'D16']].values
    
    correlations = []
    for i in range(len(dk_vecs)):
        for j in range(i+1, len(dk_vecs)):
            if not np.any(np.isnan(dk_vecs[i])) and not np.any(np.isnan(dk_vecs[j])):
                r, _ = stats.pearsonr(dk_vecs[i], dk_vecs[j])
                correlations.append(r)
    
    mean_corr = np.mean(correlations) if correlations else np.nan
    task_correlations.append(mean_corr)
    print(f"  {task}: mean r = {mean_corr:.3f}")

mean_within = np.nanmean(task_correlations)
print(f"\nMean within-task correlation: {mean_within:.3f}")

print("\n" + "=" * 70)
print("STEP 6: BETWEEN-TASK SEPARABILITY")
print("=" * 70)

pivot = df.pivot(index='subject', columns='task', values='early_auc')

between_distances = []
for subj in pivot.index:
    subj_aucs = pivot.loc[subj].values
    for i in range(len(subj_aucs)):
        for j in range(i+1, len(subj_aucs)):
            if not np.isnan(subj_aucs[i]) and not np.isnan(subj_aucs[j]):
                between_distances.append(abs(subj_aucs[i] - subj_aucs[j]))

mean_between = np.mean(between_distances)
std_between = np.std(between_distances)

print(f"Mean between-task distance: {mean_between:.4f} ± {std_between:.4f}")

print("\n" + "=" * 70)
print("STEP 7: STATISTICS")
print("=" * 70)

t_stat, p_val = stats.ttest_1samp(task_correlations, mean_between)
if np.isnan(t_stat):
    t_stat = 0
    p_val = 1

cohens_d = (mean_within - mean_between) / np.std(task_correlations) if len(task_correlations) > 1 else 0

print(f"\nMetric | Value")
print(f"-" * 30)
print(f"Within-task correlation | {mean_within:.4f}")
print(f"Between-task distance | {mean_between:.4f}")
print(f"t-statistic | {t_stat:.3f}")
print(f"p-value | {p_val:.4f}")
print(f"Cohen's d | {cohens_d:.3f}")

print("\n" + "=" * 70)
print("STEP 8: SANITY CHECKS")
print("=" * 70)

for task in TASKS:
    task_aucs = df[df.task == task]['early_auc']
    var_auc = np.var(task_aucs)
    nan_count = np.isnan(task_aucs).sum()
    print(f"  {task}: var={var_auc:.6f}, NaN={nan_count}")
    if var_auc < 1e-10:
        print(f"FAIL: {task} has near-zero variance")
        exit(1)
    if nan_count > 0:
        print(f"FAIL: {task} has NaN values")
        exit(1)

df.to_csv(os.path.join(OUT_DIR, "regime_stability_results.csv"), index=False)
print(f"\nSaved: {OUT_DIR}regime_stability_results.csv")

print("\n" + "=" * 70)
print("FINAL OUTPUT")
print("=" * 70)
print(f"\nTable:")
print(f"Metric | Value")
print(f"-" * 30)
print(f"Within-task correlation | {mean_within:.4f} ± {np.std(task_correlations):.4f}")
print(f"Between-task distance | {mean_between:.4f} ± {std_between:.4f}")
print(f"t-statistic | {t_stat:.3f}")
print(f"p-value | {p_val:.4f}")
print(f"Cohen's d | {cohens_d:.3f}")