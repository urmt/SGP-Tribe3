#!/usr/bin/env python3
"""SFH-SGP_MULTITASK - FIXED"""
import os, gc, numpy as np, pandas as pd, nibabel as nib
from scipy import stats
from sklearn.decomposition import PCA

DATA_ROOT = "/home/student/sgp-tribe3/empirical_analysis/data/ds000030_s3/"
SUBJ_CSV = "/home/student/sgp-tribe3/empirical_analysis/ds000030_qc/clean_subjects.csv"
TASKS = ['rest', 'bart', 'scap', 'stopsignal', 'taskswitch']

subj_df = pd.read_csv(SUBJ_CSV)
subject_list = subj_df['subject'].tolist()
print("Checking subjects...")

valid = []
for subj_id in subject_list:
    check = True
    for task_name in TASKS:
        if not os.path.exists(f'{DATA_ROOT}/{subj_id}/func/{subj_id}_task-{task_name}_bold.nii.gz'):
            check = False
            break
    if check:
        valid.append(subj_id)

print(f"Valid: {len(valid)}")

results = []
idx = 0
while idx < len(valid):
    subj_id = valid[idx]
    
    for task_name in TASKS:
        img_load = nib.load(f'{DATA_ROOT}/{subj_id}/func/{subj_id}_task-{task_name}_bold.nii.gz')
        arr_data = img_load.get_fdata()
        ts_data = arr_data.reshape(-1, arr_data.shape[3]).T[:50, :3000]
        del arr_data, img_load
        gc.collect()
        
        m_vals = np.mean(ts_data, axis=0)
        s_vals = np.std(ts_data, axis=0)
        s_vals[s_vals == 0] = 1
        ts_data = (ts_data - m_vals) / s_vals
        
        pca_model = PCA(3)
        pca_model.fit(ts_data)
        dim_val = np.sum(pca_model.explained_variance_ratio_[:4])
        
        results.append({'subject': subj_id, 'task': task_name, 'AUC': dim_val})
        del ts_data
        gc.collect()
    
    idx = idx + 1

df_results = pd.DataFrame(results)
pivot_table = df_results.pivot(index='subject', columns='task', values='AUC')

print("\nMeans:")
for task_name in TASKS:
    print(f"  {task_name}: {df_results[df_results.task==task_name].AUC.mean():.3f}")

stat_f, pval = stats.f_oneaway(*[df_results[df_results.task==tk].AUC for tk in TASKS])
print(f"\nANOVA: F={stat_f:.2f}, p={pval:.4f}")

ranked_tasks = df_results.groupby('task').AUC.mean().sort_values(ascending=False)
print(f"Ranked: {list(ranked_tasks.index)}")
print(f"Differs: {'YES' if pval < 0.05 else 'NO'}")