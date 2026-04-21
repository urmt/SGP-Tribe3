#!/usr/bin/env python3
"""
Representational Contrast: SWITCH vs REPEAT
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
print("REPRESENTATIONAL CONTRAST: SWITCH vs REPEAT")
print("=" * 60)

subjs = pd.read_csv(SUBJ_CSV)['subject'].tolist()
results = []
failures = []

for si, s in enumerate(subjs):
    bold_path = f"{DATA_ROOT}/{s}/func/{s}_task-taskswitch_bold.nii.gz"
    events_path = f"{DATA_ROOT}/{s}/func/{s}_task-taskswitch_events.tsv"
    
    if not os.path.exists(bold_path) or not os.path.exists(events_path):
        failures.append((s, "missing"))
        continue
    
    try:
        events = pd.read_csv(events_path, sep='\t')
        
        switch_mask = events['Switching'] == 'SWITCH'
        repeat_mask = events['Switching'] == 'NOSWITCH'
        
        switch_onsets = events.loc[switch_mask, 'onset'].values
        repeat_onsets = events.loc[repeat_mask, 'onset'].values
        
        if len(switch_onsets) < 3 or len(repeat_onsets) < 3:
            failures.append((s, "few trials"))
            continue
        
        img = nib.load(bold_path)
        arr = img.get_fdata()
        n_tp = arr.shape[3]
        del img
        
        full_ts = arr.reshape(-1, n_tp).T
        del arr
        gc.collect()
        
        switch_segs = []
        for onset in switch_onsets[:15]:
            vol = int(onset * 2)
            if vol + 30 <= n_tp:
                switch_segs.append(full_ts[vol:vol+30])
        
        repeat_segs = []
        for onset in repeat_onsets[:15]:
            vol = int(onset * 2)
            if vol + 30 <= n_tp:
                repeat_segs.append(full_ts[vol:vol+30])
        
        del full_ts
        gc.collect()
        
        if len(switch_segs) < 2 or len(repeat_segs) < 2:
            failures.append((s, "segments"))
            continue
        
        switch_ts = np.vstack(switch_segs)
        repeat_ts = np.vstack(repeat_segs)
        
        switch_ts = switch_ts[:, ::max(1, switch_ts.shape[1] // 1500)]
        repeat_ts = repeat_ts[:, ::max(1, repeat_ts.shape[1] // 1500)]
        
        for j in range(switch_ts.shape[1]):
            y = switch_ts[:, j]
            switch_ts[:, j] = (y - np.mean(y)) / (np.std(y) + 1e-10)
        
        for j in range(repeat_ts.shape[1]):
            y = repeat_ts[:, j]
            repeat_ts[:, j] = (y - np.mean(y)) / (np.std(y) + 1e-10)
        
        d2_sw = compute_dimensionality(switch_ts, 2)
        d4_sw = compute_dimensionality(switch_ts, 4)
        auc_switch = np.nanmean([d2_sw, d4_sw])
        
        d2_rep = compute_dimensionality(repeat_ts, 2)
        d4_rep = compute_dimensionality(repeat_ts, 4)
        auc_repeat = np.nanmean([d2_rep, d4_rep])
        
        if not np.isnan(auc_switch) and not np.isnan(auc_repeat):
            results.append({
                'subject': s,
                'auc_switch': auc_switch,
                'auc_repeat': auc_repeat,
                'diff': auc_switch - auc_repeat
            })
            print(f"[{si+1}] {s}: SWITCH={auc_switch:.3f} REPEAT={auc_repeat:.3f}")
        
        del switch_ts, repeat_ts
        gc.collect()
        
    except Exception as e:
        failures.append((s, str(e)))
        print(f"[{si+1}] {s}: ERROR {e}")

print(f"\nValid: {len(results)}, Failed: {len(failures)}")

if len(results) < 10:
    print("FAIL: Too few subjects")
    exit(1)

df = pd.DataFrame(results)

mean_switch = df['auc_switch'].mean()
mean_repeat = df['auc_repeat'].mean()
mean_diff = df['diff'].mean()

print(f"\nMean AUC:")
print(f"  SWITCH: {mean_switch:.4f}")
print(f"  REPEAT: {mean_repeat:.4f}")
print(f"  Diff: {mean_diff:+.4f}")

t_stat, p_val = stats.ttest_rel(df['auc_switch'], df['auc_repeat'])
pooled_std = np.sqrt((df['auc_switch'].var() + df['auc_repeat'].var()) / 2)
cohens_d = mean_diff / pooled_std

print(f"\nPaired t-test: t={t_stat:.3f}, p={p_val:.4f}, d={cohens_d:.3f}")

df.to_csv(OUT + "switch_repeat_contrast.csv", index=False)

with open(OUT + "switch_repeat_stats.txt", "w") as f:
    f.write("REPRESENTATIONAL CONTRAST: SWITCH vs REPEAT\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Mean AUC:\n  SWITCH: {mean_switch:.4f}\n  REPEAT: {mean_repeat:.4f}\n  Diff: {mean_diff:+.4f}\n\n")
    f.write(f"t-test: t={t_stat:.3f}, p={p_val:.4f}, d={cohens_d:.3f}\n")

result = "YES" if p_val < 0.05 and abs(cohens_d) > 0.2 else "NO"
print(f"\nDimensionality differs: {result}")