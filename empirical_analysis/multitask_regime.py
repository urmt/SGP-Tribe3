#!/usr/bin/env python3
"""
SFH-SGP_MULTITASK_REGIME_MAPPING_01 - MINIMAL
"""
import os, gc, numpy as np, pandas as pd, nibabel as nib
from scipy import stats
from sklearn.decomposition import PCA

DATA_ROOT = "/home/student/sgp-tribe3/empirical_analysis/data/ds000030_s3/"
SUBJ_CSV = "/home/student/sgp-tribe3/empirical_analysis/ds000030_qc/clean_subjects.csv"
OUT = "/home/student/sgp-tribe3/empirical_analysis/multitask_results/"

TASKS = ['rest', 'bart', 'scap', 'stopsignal', 'taskswitch']

print("VALIDATION & ANALYSIS")

subjs = pd.read_csv(SUBJ_CSV)['subject'].tolist()
valid = [s for s in subjs if all(os.path.exists(f'{DATA_ROOT}/{s}/func/{s}_task-{t}_bold.nii.gz') for t in TASKS)]
print(f"Valid: {len(valid)}")

results = []
for s in valid:
    for task in TASKS:
        arr = nib.load(f"{DATA_ROOT}/{s}/func/{s}_task-{task}_bold.nii.gz").get_fdata()
        ts = arr.reshape(-1, arr.shape[3]).T.astype(np.float32)[:50, :5000]
        del arr; gc.collect()
        for j in range(ts.shape[1]):
            m, std = np.mean(ts[:,j]), np.std(ts[:,j])
            ts[:,j] = (ts[:,j]-m)/(std+1e-10) if std>1e-10 else 0
        p = PCA(5); p.fit(ts); d = np.sum(p.explained_variance_ratio_[:4])
        results.append({'subject':s, 'task':task, 'AUC':d})
        del ts; gc.collect()

df = pd.DataFrame(results)
pivot = df.pivot(index='subject', columns='task', values='AUC')

print("\nMeans:")
for t in TASKS:
    print(f"  {t}: {df[df.task==t].AUC.mean():.3f}")

f, p = stats.f_oneway(*[df[df.task==t].AUC for t in TASKS])
print(f"\nANOVA: F={f:.2f}, p={p:.4f}")

ranked = df.groupby('task').AUC.mean().sort_values(ascending=False)
print(f"Ranked: {list(ranked.index)}")
print(f"Highest: {ranked.index[0]}, Lowest: {ranked.index[-1]}")
print(f"Dimensionality differs: {'YES' if p < 0.05 else 'NO'}")

df.to_csv(OUT + "multitask_results.csv", index=False)