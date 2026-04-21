#!/usr/bin/env python3
"""
SFH-SGP_REGIME_STABILITY_SHUFFLE_TEST
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
print("LOAD SUBJECTS")
print("=" * 70)

subj_list = pd.read_csv(SUBJ_CSV)['subject'].tolist()[:MAX_SUBJECTS]
print(f"Using {len(subj_list)} subjects")

print("\n" + "=" * 70)
print("COMPUTE D(k) VECTORS")
print("=" * 70)

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

df_real = pd.DataFrame(results)

dk_cols = ['D2', 'D4', 'D8', 'D16']

print("\n" + "=" * 70)
print("REAL DATA: SANITY CHECK - SAMPLE D(k) VECTORS")
print("=" * 70)

for subj in subj_list[:3]:
    print(f"\n{subj}:")
    for task in TASKS:
        row = df_real[(df_real.subject == subj) & (df_real.task == task)].iloc[0]
        vec = [row['D2'], row['D4'], row['D8'], row['D16']]
        print(f"  {task:12s}: [{vec[0]:.4f}, {vec[1]:.4f}, {vec[2]:.4f}, {vec[3]:.4f}]")

nan_count = df_real[dk_cols].isna().sum().sum()
print(f"\nNaN count: {nan_count}")
if nan_count > 0:
    print("FAIL: NaN detected")
    exit(1)

print("\n" + "=" * 70)
print("REAL: COMPUTE WITHIN-TASK CORRELATIONS")
print("=" * 70)

within_task_corrs_real = []
for task in TASKS:
    task_df = df_real[df_real.task == task]
    dk_matrix = task_df[dk_cols].values
    
    corr_vals = []
    for i in range(len(dk_matrix)):
        for j in range(i+1, len(dk_matrix)):
            if not np.any(np.isnan(dk_matrix[i])) and not np.any(np.isnan(dk_matrix[j])):
                r, _ = stats.pearsonr(dk_matrix[i], dk_matrix[j])
                corr_vals.append(r)
    
    mean_corr = np.mean(corr_vals) if corr_vals else np.nan
    within_task_corrs_real.extend(corr_vals)
    print(f"  {task:12s}: mean r = {mean_corr:.4f}")

mean_within_real = np.mean(within_task_corrs_real)
print(f"\n  REAL mean within-task correlation: {mean_within_real:.4f}")

print("\n" + "=" * 70)
print("REAL: COMPUTE BETWEEN-TASK DISTANCES")
print("=" * 70)

between_task_dists_real = []
for subj in subj_list:
    subj_data = df_real[df_real.subject == subj]
    
    for i, t1 in enumerate(TASKS):
        row1 = subj_data[subj_data.task == t1][dk_cols].values
        if len(row1) == 0:
            continue
        vec1 = row1[0]
        
        for t2 in TASKS[i+1:]:
            row2 = subj_data[subj_data.task == t2][dk_cols].values
            if len(row2) == 0:
                continue
            vec2 = row2[0]
            
            if not np.any(np.isnan(vec1)) and not np.any(np.isnan(vec2)):
                dist = np.sqrt(np.sum((vec1 - vec2)**2))
                between_task_dists_real.append(dist)

mean_between_real = np.mean(between_task_dists_real)
print(f"  REAL mean between-task distance: {mean_between_real:.4f}")

df_real.to_csv(os.path.join(OUT_DIR, "regime_real.csv"), index=False)
print(f"\nSaved: {OUT_DIR}regime_real.csv")

print("\n" + "=" * 70)
print("STEP 2: SHUFFLE TASK LABELS")
print("=" * 70)

np.random.seed(42)

df_shuffled = df_real.copy()

for subj in subj_list:
    subj_mask = df_shuffled['subject'] == subj
    task_labels = df_shuffled.loc[subj_mask, 'task'].values
    shuffled_labels = np.random.permutation(task_labels)
    df_shuffled.loc[subj_mask, 'task'] = shuffled_labels

print("Shuffled task assignments per subject:")
for subj in subj_list[:3]:
    real_tasks = df_real[df_real.subject == subj]['task'].tolist()
    shuff_tasks = df_shuffled[df_shuffled.subject == subj]['task'].tolist()
    print(f"  {subj}:")
    print(f"    real:    {real_tasks}")
    print(f"    shuffled: {shuff_tasks}")

print("\n" + "=" * 70)
print("SHUFFLED: COMPUTE WITHIN-TASK CORRELATIONS")
print("=" * 70)

within_task_corrs_shuffled = []
for task in TASKS:
    task_df = df_shuffled[df_shuffled.task == task]
    dk_matrix = task_df[dk_cols].values
    
    corr_vals = []
    for i in range(len(dk_matrix)):
        for j in range(i+1, len(dk_matrix)):
            if not np.any(np.isnan(dk_matrix[i])) and not np.any(np.isnan(dk_matrix[j])):
                r, _ = stats.pearsonr(dk_matrix[i], dk_matrix[j])
                corr_vals.append(r)
    
    mean_corr = np.mean(corr_vals) if corr_vals else np.nan
    within_task_corrs_shuffled.extend(corr_vals)
    print(f"  {task:12s}: mean r = {mean_corr:.4f}")

mean_within_shuffled = np.mean(within_task_corrs_shuffled)
print(f"\n  SHUFFLED mean within-task correlation: {mean_within_shuffled:.4f}")

print("\n" + "=" * 70)
print("SHUFFLED: COMPUTE BETWEEN-TASK DISTANCES")
print("=" * 70)

between_task_dists_shuffled = []
for subj in subj_list:
    subj_data = df_shuffled[df_shuffled.subject == subj]
    
    for i, t1 in enumerate(TASKS):
        row1 = subj_data[subj_data.task == t1][dk_cols].values
        if len(row1) == 0:
            continue
        vec1 = row1[0]
        
        for t2 in TASKS[i+1:]:
            row2 = subj_data[subj_data.task == t2][dk_cols].values
            if len(row2) == 0:
                continue
            vec2 = row2[0]
            
            if not np.any(np.isnan(vec1)) and not np.any(np.isnan(vec2)):
                dist = np.sqrt(np.sum((vec1 - vec2)**2))
                between_task_dists_shuffled.append(dist)

mean_between_shuffled = np.mean(between_task_dists_shuffled)
print(f"  SHUFFLED mean between-task distance: {mean_between_shuffled:.4f}")

df_shuffled.to_csv(os.path.join(OUT_DIR, "regime_shuffled.csv"), index=False)
print(f"\nSaved: {OUT_DIR}regime_shuffled.csv")

print("\n" + "=" * 70)
print("STEP 4: COMPARE REAL vs SHUFFLED")
print("=" * 70)

print(f"\n{'Metric':<35} {'REAL':<15} {'SHUFFLED':<15}")
print("-" * 65)
print(f"{'Mean within-task correlation':<35} {mean_within_real:<15.4f} {mean_within_shuffled:<15.4f}")
print(f"{'Mean between-task distance':<35} {mean_between_real:<15.4f} {mean_between_shuffled:<15.4f}")

t_within, p_within = stats.ttest_ind(within_task_corrs_real, within_task_corrs_shuffled)
t_between, p_between = stats.ttest_ind(between_task_dists_real, between_task_dists_shuffled)

print(f"\nStatistical tests:")
print(f"  Within-task: t = {t_within:.4f}, p = {p_within:.6f}")
print(f"  Between-task: t = {t_between:.4f}, p = {p_between:.6f}")

print("\n" + "=" * 70)
print("STEP 5: INTERPRETATION")
print("=" * 70)

if abs(mean_within_real - mean_within_shuffled) < 0.01 and abs(mean_between_real - mean_between_shuffled) < 0.01:
    print("RESULT: FAIL (artifact detected - real ≈ shuffled)")
else:
    print("RESULT: PASS (real signal detected)")
    if mean_within_real > mean_within_shuffled:
        print("  - Within-task correlation: REAL > SHUFFLED")
    if mean_between_real < mean_between_shuffled:
        print("  - Between-task distance: REAL < SHUFFLED")
