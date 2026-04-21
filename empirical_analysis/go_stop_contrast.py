#!/usr/bin/env python3
"""
Representational Contrast: GO vs STOP (fixed)
"""
import os, gc, numpy as np, pandas as pd, nibabel as nib
from scipy import stats
from sklearn.decomposition import PCA

DATA_ROOT = "/home/student/sgp-tribe3/empirical_analysis/data/ds000030_s3/"
SUBJ_CSV = "/home/student/sgp-tribe3/empirical_analysis/ds000030_qc/clean_subjects.csv"
OUT = "/home/student/sgp-tribe3/empirical_analysis/multitask_results/"

def compute_dimensionality(data, k):
    try:
        if data.shape[0] < k + 2:
            return np.nan
        p = PCA(min(k + 1, data.shape[0] - 1))
        p.fit(data)
        return np.sum(p.explained_variance_ratio_[:k])
    except:
        return np.nan

print("=" * 60)
print("REPRESENTATIONAL CONTRAST: GO vs STOP")
print("=" * 60)

subjs = pd.read_csv(SUBJ_CSV)['subject'].tolist()
results = []
failures = []

for si, s in enumerate(subjs):
    bold_path = f"{DATA_ROOT}/{s}/func/{s}_task-stopsignal_bold.nii.gz"
    events_path = f"/home/student/sgp-tribe3/empirical_analysis/data/ds000030_s3/{s}/func/{s}_task-stopsignal_events.tsv"
    
    if not os.path.exists(bold_path) or not os.path.exists(events_path):
        failures.append((s, "missing"))
        continue
    
    try:
        events = pd.read_csv(events_path, sep='\t')
        
        go_mask = events['trial_type'] == 'GO'
        stop_mask = events['trial_type'] == 'STOP'
        
        go_onsets = events.loc[go_mask, 'onset'].values
        stop_onsets = events.loc[stop_mask, 'onset'].values
        
        if len(go_onsets) < 3 or len(stop_onsets) < 2:
            failures.append((s, "few trials"))
            continue
        
        img = nib.load(bold_path)
        arr = img.get_fdata()
        n_tp = arr.shape[3]
        del img
        
        full_ts = arr.reshape(-1, n_tp).T
        del arr
        gc.collect()
        
        go_segs = []
        for onset in go_onsets[:15]:
            vol = int(onset * 2)
            if vol + 30 <= n_tp:
                go_segs.append(full_ts[vol:vol+30])
        
        stop_segs = []
        for onset in stop_onsets[:10]:
            vol = int(onset * 2)
            if vol + 30 <= n_tp:
                stop_segs.append(full_ts[vol:vol+30])
        
        del full_ts
        gc.collect()
        
        if len(go_segs) < 2 or len(stop_segs) < 1:
            failures.append((s, "segments"))
            continue
        
        go_ts = np.vstack(go_segs)
        stop_ts = np.vstack(stop_segs)
        
        go_ts = go_ts[:, ::max(1, go_ts.shape[1] // 1500)]
        stop_ts = stop_ts[:, ::max(1, stop_ts.shape[1] // 1500)]
        
        for j in range(go_ts.shape[1]):
            y = go_ts[:, j]
            go_ts[:, j] = (y - np.mean(y)) / (np.std(y) + 1e-10)
        
        for j in range(stop_ts.shape[1]):
            y = stop_ts[:, j]
            stop_ts[:, j] = (y - np.mean(y)) / (np.std(y) + 1e-10)
        
        d2_go = compute_dimensionality(go_ts, 2)
        d4_go = compute_dimensionality(go_ts, 4)
        auc_go = np.nanmean([d2_go, d4_go])
        
        d2_stop = compute_dimensionality(stop_ts, 2)
        d4_stop = compute_dimensionality(stop_ts, 4)
        auc_stop = np.nanmean([d2_stop, d4_stop])
        
        if not np.isnan(auc_go) and not np.isnan(auc_stop):
            results.append({
                'subject': s,
                'auc_go': auc_go,
                'auc_stop': auc_stop,
                'diff': auc_go - auc_stop
            })
            print(f"[{si+1}] {s}: GO={auc_go:.3f} STOP={auc_stop:.3f}")
        
        del go_ts, stop_ts
        gc.collect()
        
    except Exception as e:
        failures.append((s, str(e)))
        print(f"[{si+1}] {s}: ERROR {e}")

print(f"\nValid: {len(results)}, Failed: {len(failures)}")

if len(results) < 10:
    print("FAIL: Too few subjects")
    exit(1)

df = pd.DataFrame(results)

mean_go = df['auc_go'].mean()
mean_stop = df['auc_stop'].mean()
mean_diff = df['diff'].mean()

print(f"\nMean AUC:")
print(f"  GO:   {mean_go:.4f}")
print(f"  STOP: {mean_stop:.4f}")
print(f"  Diff: {mean_diff:+.4f}")

t_stat, p_val = stats.ttest_rel(df['auc_go'], df['auc_stop'])
pooled_std = np.sqrt((df['auc_go'].var() + df['auc_stop'].var()) / 2)
cohens_d = mean_diff / pooled_std

print(f"\nPaired t-test: t={t_stat:.3f}, p={p_val:.4f}, d={cohens_d:.3f}")

df.to_csv(OUT + "go_stop_contrast.csv", index=False)

with open(OUT + "representational_contrast_stats.txt", "w") as f:
    f.write("REPRESENTATIONAL CONTRAST: GO vs STOP\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Mean AUC:\n  GO: {mean_go:.4f}\n  STOP: {mean_stop:.4f}\n  Diff: {mean_diff:+.4f}\n\n")
    f.write(f"t-test: t={t_stat:.3f}, p={p_val:.4f}, d={cohens_d:.3f}\n")

result = "YES" if p_val < 0.05 and abs(cohens_d) > 0.2 else "NO"
print(f"\nDimensionality differs: {result}")