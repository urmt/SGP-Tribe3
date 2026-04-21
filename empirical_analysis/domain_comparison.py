#!/usr/bin/env python3
"""
SFH-SGP_DOMAIN_REGIME_COMPARISON_01
Compare ds000114 (motor/cognitive) vs ds000030 (cognitive tasks)
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats
from sklearn.neighbors import NearestNeighbors

DATA1 = "/home/student/sgp-tribe3/empirical_analysis/data/ds000114_full/"
DATA2 = "/home/student/sgp-tribe3/empirical_analysis/data/ds000030_s3/"
OUT = "/home/student/sgp-tribe3/empirical_analysis/multitask_results/"

MAX_VOXELS = 3000

def compute_knn_dim(data, k):
    try:
        n, d = data.shape
        if n <= k + 1 or d < 10:
            return np.nan
        nn = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree')
        nn.fit(data)
        dists, _ = nn.kneighbors(data)
        log_d = np.log(dists[:, -1] + 1e-10)
        return np.mean(log_d)
    except:
        return np.nan

def process_subject(subject_path, task_name):
    """Process single task for a subject"""
    img = nib.load(subject_path)
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
    del ts
    
    if np.isnan(d2) or np.isnan(d4):
        return np.nan
    return (d2 + d4) / 2

print("=" * 70)
print("DOMAIN REGIME COMPARISON")
print("=" * 70)

results = []

# ds000114: motor task (fingerfootlips) vs rest
print("\n--- ds000114 (Motor/Cognitive) ---")

sub114 = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05']

for subj in sub114:
    ses_path = f"{DATA1}/{subj}/ses-test/func/"
    
    motor_path = f"{ses_path}{subj}_ses-test_task-fingerfootlips_bold.nii.gz"
    rest_path = f"{ses_path}{subj}_ses-test_task-linebisection_bold.nii.gz"  # Using linebis as rest-like
    
    print(f"{subj}:", end=" ")
    
    try:
        auc_motor = process_subject(motor_path, 'motor')
        print(f"motor={auc_motor:.3f}", end=" ")
    except:
        auc_motor = np.nan
        print("motor=ERR", end=" ")
    
    try:
        auc_rest = process_subject(rest_path, 'rest')
        print(f"rest={auc_rest:.3f}")
    except:
        auc_rest = np.nan
        print("rest=ERR")
    
    results.append({'dataset': 'ds000114', 'subject': subj, 'condition': 'motor', 'AUC': auc_motor})
    results.append({'dataset': 'ds000114', 'subject': subj, 'condition': 'rest', 'AUC': auc_rest})

# ds000030: cognitive tasks vs rest
print("\n--- ds000030 (Cognitive) ---")

subj_list = pd.read_csv("/home/student/sgp-tribe3/empirical_analysis/ds000030_qc/clean_subjects.csv")['subject'].tolist()[:10]

for subj in subj_list:
    base = f"{DATA2}/{subj}/func/"
    
    task_path = f"{base}{subj}_task-bart_bold.nii.gz"
    rest_path = f"{base}{subj}_task-rest_bold.nii.gz"
    
    print(f"{subj}:", end=" ")
    
    try:
        auc_task = process_subject(task_path, 'task')
        print(f"task={auc_task:.3f}", end=" ")
    except Exception as e:
        auc_task = np.nan
        print(f"task=ERR", end=" ")
    
    try:
        auc_rest = process_subject(rest_path, 'rest')
        print(f"rest={auc_rest:.3f}")
    except:
        auc_rest = np.nan
        print("rest=ERR")
    
    results.append({'dataset': 'ds000030', 'subject': subj, 'condition': 'task', 'AUC': auc_task})
    results.append({'dataset': 'ds000030', 'subject': subj, 'condition': 'rest', 'AUC': auc_rest})

df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

# ds000114 comparison
ds114 = df[df.dataset == 'ds000114']
motor_auc = ds114[ds114.condition == 'motor']['AUC'].dropna()
rest_auc_114 = ds114[ds114.condition == 'rest']['AUC'].dropna()

print(f"\nds000114 (Motor/Cognitive):")
print(f"  Motor: {motor_auc.mean():.4f} ± {motor_auc.std():.4f}")
print(f"  Rest-like: {rest_auc_114.mean():.4f} ± {rest_auc_114.std():.4f}")

if len(motor_auc) > 1 and len(rest_auc_114) > 1:
    t114, p114 = stats.ttest_rel(motor_auc, rest_auc_114)
    d114 = (motor_auc.mean() - rest_auc_114.mean()) / motor_auc.std()
    print(f"  t={t114:.2f}, p={p114:.4f}, d={d114:.2f}")

# ds000030 comparison
ds030 = df[df.dataset == 'ds000030']
task_auc = ds030[ds030.condition == 'task']['AUC'].dropna()
rest_auc_030 = ds030[ds030.condition == 'rest']['AUC'].dropna()

print(f"\nds000030 (Cognitive):")
print(f"  Task: {task_auc.mean():.4f} ± {task_auc.std():.4f}")
print(f"  Rest: {rest_auc_030.mean():.4f} ± {rest_auc_030.std():.4f}")

if len(task_auc) > 1 and len(rest_auc_030) > 1:
    t030, p030 = stats.ttest_rel(task_auc, rest_auc_030)
    d030 = (task_auc.mean() - rest_auc_030.mean()) / task_auc.std()
    print(f"  t={t030:.2f}, p={p030:.4f}, d={d030:.2f}")

# Cross-dataset
print(f"\n--- Cross-Dataset ---")
print(f"ds000114 motor vs ds000030 task:")
t_cross, p_cross = stats.ttest_ind(motor_auc, task_auc)
d_cross = (motor_auc.mean() - task_auc.mean()) / np.sqrt((motor_auc.std()**2 + task_auc.std()**2)/2)
print(f"  t={t_cross:.2f}, p={p_cross:.4f}, d={d_cross:.2f}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if p114 < 0.05 and p030 < 0.05:
    if abs(d114) > abs(d030):
        print("Dimensionality differs MORE within ds000114 (domain effect)")
    else:
        print("Dimensionality differs MORE within ds000030 (task effect)")
elif p_cross < 0.05:
    print("Significant domain difference between datasets")
else:
    print("No significant differences found")

df.to_csv(OUT + "domain_comparison_results.csv", index=False)