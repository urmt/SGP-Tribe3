#!/usr/bin/env python3
"""
SFH-SGP_REGIME_STABILITY_AND_SEPARABILITY_01_REDO
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
print("STEP 1: LOAD SUBJECTS")
print("=" * 70)

subj_list = pd.read_csv(SUBJ_CSV)['subject'].tolist()[:MAX_SUBJECTS]
print(f"Using {len(subj_list)} subjects")

print("\n" + "=" * 70)
print("STEP 2: COMPUTE D(k) VECTORS FOR ALL SUBJECT × TASK")
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

df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("STEP 3: SANITY CHECK")
print("=" * 70)

print("\nSample D(k) vectors for 3 subjects:")
for subj in subj_list[:3]:
    print(f"\n{subj}:")
    for task in TASKS:
        row = df[(df.subject == subj) & (df.task == task)].iloc[0]
        vec = [row['D2'], row['D4'], row['D8'], row['D16']]
        print(f"  {task:12s}: D2={vec[0]:.4f}, D4={vec[1]:.4f}, D8={vec[2]:.4f}, D16={vec[3]:.4f}")

nan_count = df[['D2','D4','D8','D16']].isna().sum().sum()
print(f"\nTotal NaN count: {nan_count}")

for k in K_VALUES:
    col = f'D{k}'
    std_val = df[col].std()
    print(f"D{k} std across all: {std_val:.6f}")

print("\n" + "=" * 70)
print("STEP 4: WITHIN-TASK CONSISTENCY")
print("=" * 70)

dk_cols = ['D2', 'D4', 'D8', 'D16']

within_task_corrs = []
for task in TASKS:
    task_df = df[df.task == task]
    dk_matrix = task_df[dk_cols].values
    
    corr_vals = []
    for i in range(len(dk_matrix)):
        for j in range(i+1, len(dk_matrix)):
            if not np.any(np.isnan(dk_matrix[i])) and not np.any(np.isnan(dk_matrix[j])):
                r, _ = stats.pearsonr(dk_matrix[i], dk_matrix[j])
                corr_vals.append(r)
    
    mean_corr = np.mean(corr_vals) if corr_vals else np.nan
    std_corr = np.std(corr_vals) if corr_vals else np.nan
    within_task_corrs.extend(corr_vals)
    print(f"  {task:12s}: mean r = {mean_corr:.4f} ± {std_corr:.4f} (n={len(corr_vals)})")

mean_within = np.mean(within_task_corrs)
std_within = np.std(within_task_corrs)
print(f"\n  OVERALL: mean r = {mean_within:.4f} ± {std_within:.4f}")

print("\n" + "=" * 70)
print("STEP 5: BETWEEN-TASK SEPARABILITY")
print("=" * 70)

dk_cols = ['D2', 'D4', 'D8', 'D16']

between_task_dists = []
for subj in subj_list:
    subj_data = df[df.subject == subj]
    
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
                between_task_dists.append(dist)

mean_between = np.mean(between_task_dists)
std_between = np.std(between_task_dists)
print(f"  Mean Euclidean distance: {mean_between:.4f} ± {std_between:.4f} (n={len(between_task_dists)})")

print("\n" + "=" * 70)
print("STEP 6: STATISTICS")
print("=" * 70)

t_stat, p_val = stats.ttest_1samp(within_task_corrs, mean_between)
cohens_d = (mean_within - mean_between) / np.std(within_task_corrs) if len(within_task_corrs) > 1 else np.nan

print(f"\n{'Metric':<30} {'Value'}")
print("-" * 45)
print(f"{'Within-task correlation':<30} {mean_within:.4f} ± {std_within:.4f}")
print(f"{'Between-task distance':<30} {mean_between:.4f} ± {std_between:.4f}")
print(f"{'t-statistic':<30} {t_stat:.4f}")
print(f"{'p-value':<30} {p_val:.6f}")
print(f"{'Cohens d':<30} {cohens_d:.4f}")

print("\n" + "=" * 70)
print("FINAL OUTPUT")
print("=" * 70)
print(f"\n{'Metric':<30} {'Value'}")
print("-" * 45)
print(f"{'Within-task correlation':<30} {mean_within:.4f} ± {std_within:.4f}")
print(f"{'Between-task distance':<30} {mean_between:.4f} ± {std_between:.4f}")
print(f"{'t-statistic':<30} {t_stat:.4f}")
print(f"{'p-value':<30} {p_val:.6f}")
print(f"{'Cohens d':<30} {cohens_d:.4f}")

df.to_csv(os.path.join(OUT_DIR, "regime_stability_dk_vectors.csv"), index=False)
print(f"\nSaved D(k) vectors to: {OUT_DIR}regime_stability_dk_vectors.csv")
