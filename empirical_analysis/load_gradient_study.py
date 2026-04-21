#!/usr/bin/env python3
"""
Load Gradient Study - Chunked with checkpointing
"""

import os
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats
from sklearn.decomposition import PCA
import warnings
import gc
warnings.filterwarnings('ignore')

DATA_ROOT = "/home/student/sgp-tribe3/empirical_analysis/data/ds000030_s3/"
SUBJECTS_FILE = "/home/student/sgp-tribe3/empirical_analysis/ds000030_qc/clean_subjects.csv"
OUTPUT_DIR = "/home/student/sgp-tribe3/empirical_analysis/multitask_results/"
CHECKPOINT = os.path.join(OUTPUT_DIR, "checkpoint.pkl")

TASKS = {'rest': 1, 'scap': 2, 'bart': 3, 'stopsignal': 4, 'taskswitch': 5}

def compute_dim(data, k):
    try:
        if data.shape[0] < 15:
            return np.nan
        pca = PCA(n_components=min(k+1, data.shape[0]-1))
        pca.fit(data)
        var = pca.explained_variance_ratio_
        return np.nansum(var[:k])
    except:
        return np.nan

def load_process(path):
    if not os.path.exists(path):
        return None
    try:
        img = nib.load(path)
        arr = img.get_fdata(dt=np.float32)
        del img
        shape = arr.shape
        ts = arr.reshape(-1, shape[3]).T
        del arr; gc.collect()
        
        if ts.shape[0] < 25:
            return None
        
        ts = ts[:, ::max(1, ts.shape[1]//1500)]
        
        for j in range(ts.shape[1]):
            y = ts[:, j]
            y = y - np.polyval(np.polyfit(np.arange(len(y)), y, 1), np.arange(len(y)))
            ts[:, j] = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-10)
        
        d2, d4 = compute_dim(ts, 2), compute_dim(ts, 4)
        del ts; gc.collect()
        
        return {'D_2': d2, 'D_4': d4, 'AUC': np.nanmean([d2, d4])}
    except Exception as e:
        return None

def main():
    print("STUDY 02 REAL", flush=True)
    
    df_subj = pd.read_csv(SUBJECTS_FILE)
    subjects = df_subj['subject'].tolist()
    
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, 'rb') as f:
            results = pickle.load(f)
        start_idx = len(results) // 5
        print(f"Resuming from {start_idx}", flush=True)
    else:
        results = []
        start_idx = 0
    
    for i in range(start_idx, len(subjects)):
        subj = subjects[i]
        print(f"[{i+1}/{len(subjects)}] {subj}", end=" ", flush=True)
        
        for task in TASKS:
            path = os.path.join(DATA_ROOT, subj, "func", f"{subj}_task-{task}_bold.nii.gz")
            res = load_process(path)
            
            if res is None:
                print("X", end="", flush=True)
                continue
            
            results.append({
                'subject': subj, 'task': task, 'load_level': TASKS[task],
                'D_2': res['D_2'], 'D_4': res['D_4'], 'AUC': res['AUC']
            })
            print(".", end="", flush=True)
        print(flush=True)
        
        if (i + 1) % 5 == 0:
            with open(CHECKPOINT, 'wb') as f:
                pickle.dump(results, f)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "auc_by_subject.csv"), index=False)
    
    slope, intercept, r, p, se = stats.linregress(results_df['load_level'], results_df['AUC'])
    
    with open(os.path.join(OUTPUT_DIR, "load_gradient_stats.txt"), "w") as f:
        f.write(f"Slope: {slope:.5f}\n")
        f.write(f"r: {r:.4f}\n")
        f.write(f"r²: {r**2:.4f}\n")
        f.write(f"p: {p:.6f}\n\n")
        
        rest = results_df[results_df['task'] == 'rest']['AUC']
        for task in ['scap', 'bart', 'stopsignal', 'taskswitch']:
            t_dat = results_df[results_df['task'] == task]['AUC']
            t, pv = stats.ttest_rel(t_dat, rest)
            d = (t_dat.mean() - rest.mean()) / np.sqrt((t_dat.var() + rest.var()) / 2)
            f.write(f"{task}: t={t:.2f}, p={pv:.4f}, d={d:.2f}\n")
    
    print(f"\nSlope={slope:.4f}, r={r:.3f}, p={p:.4f}")
    print("SUCCESS" if p < 0.05 and slope > 0 else "INCONCLUSIVE")
    
    try:
        os.remove(CHECKPOINT)
    except:
        pass

if __name__ == "__main__":
    main()