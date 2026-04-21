#!/usr/bin/env python3
"""
Representational Contrast: SUCCESSFUL vs FAILED STOP trials (conflict outcome)
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
print("CONFLICT INTENSITY: SUCCESS vs FAILED STOP")
print("=" * 60)

subjs = pd.read_csv(SUBJ_CSV)['subject'].tolist()
results = []
failures = []

for si, s in enumerate(subjs):
    bold_path = f"{DATA_ROOT}/{s}/func/{s}_task-stopsignal_bold.nii.gz"
    events_path = f"{DATA_ROOT}/{s}/func/{s}_task-stopsignal_events.tsv"
    
    if not os.path.exists(bold_path) or not os.path.exists(events_path):
        failures.append((s, "missing"))
        continue
    
    try:
        events = pd.read_csv(events_path, sep='\t')
        
        # SuccessfulStop = failed to stop (inhibit failure = conflict failure)
        # UnsuccessfulStop = successfully stopped (inhibit success)
        succ_mask = events['TrialOutcome'] == 'UnsuccessfulStop'
        fail_mask = events['TrialOutcome'] == 'SuccessfulStop'
        
        succ_onsets = events.loc[succ_mask, 'onset'].values
        fail_onsets = events.loc[fail_mask, 'onset'].values
        
        if len(succ_onsets) < 2 or len(fail_onsets) < 2:
            failures.append((s, "few trials"))
            continue
        
        img = nib.load(bold_path)
        arr = img.get_fdata()
        n_tp = arr.shape[3]
        del img
        
        full_ts = arr.reshape(-1, n_tp).T
        del arr
        gc.collect()
        
        succ_segs = []
        for onset in succ_onsets[:12]:
            vol = int(onset * 2)
            if vol + 30 <= n_tp:
                succ_segs.append(full_ts[vol:vol+30])
        
        fail_segs = []
        for onset in fail_onsets[:12]:
            vol = int(onset * 2)
            if vol + 30 <= n_tp:
                fail_segs.append(full_ts[vol:vol+30])
        
        del full_ts
        gc.collect()
        
        if len(succ_segs) < 1 or len(fail_segs) < 1:
            failures.append((s, "segments"))
            continue
        
        succ_ts = np.vstack(succ_segs)
        fail_ts = np.vstack(fail_segs)
        
        succ_ts = succ_ts[:, ::max(1, succ_ts.shape[1] // 1500)]
        fail_ts = fail_ts[:, ::max(1, fail_ts.shape[1] // 1500)]
        
        for j in range(succ_ts.shape[1]):
            y = succ_ts[:, j]
            succ_ts[:, j] = (y - np.mean(y)) / (np.std(y) + 1e-10)
        
        for j in range(fail_ts.shape[1]):
            y = fail_ts[:, j]
            fail_ts[:, j] = (y - np.mean(y)) / (np.std(y) + 1e-10)
        
        d2_s = compute_dimensionality(succ_ts, 2)
        d4_s = compute_dimensionality(succ_ts, 4)
        auc_succ = np.nanmean([d2_s, d4_s])
        
        d2_f = compute_dimensionality(fail_ts, 2)
        d4_f = compute_dimensionality(fail_ts, 4)
        auc_fail = np.nanmean([d2_f, d4_f])
        
        if not np.isnan(auc_succ) and not np.isnan(auc_fail):
            results.append({
                'subject': s,
                'auc_successful_stop': auc_succ,
                'auc_failed_stop': auc_fail,
                'diff': auc_succ - auc_fail
            })
            print(f"[{si+1}] {s}: SUCC={auc_succ:.3f} FAIL={auc_fail:.3f}")
        
        del succ_ts, fail_ts
        gc.collect()
        
    except Exception as e:
        failures.append((s, str(e)))
        print(f"[{si+1}] {s}: ERROR {e}")

print(f"\nValid: {len(results)}, Failed: {len(failures)}")

if len(results) < 10:
    print("FAIL: Too few subjects")
    exit(1)

df = pd.DataFrame(results)

mean_succ = df['auc_successful_stop'].mean()
mean_fail = df['auc_failed_stop'].mean()
mean_diff = df['diff'].mean()

print(f"\nMean AUC:")
print(f"  SUCCESSFUL STOP: {mean_succ:.4f}")
print(f"  FAILED STOP: {mean_fail:.4f}")
print(f"  Diff: {mean_diff:+.4f}")

t_stat, p_val = stats.ttest_rel(df['auc_successful_stop'], df['auc_failed_stop'])
pooled_std = np.sqrt((df['auc_successful_stop'].var() + df['auc_failed_stop'].var()) / 2)
cohens_d = mean_diff / pooled_std

print(f"\nPaired t-test: t={t_stat:.3f}, p={p_val:.4f}, d={cohens_d:.3f}")

df.to_csv(OUT + "stop_conflict_contrast.csv", index=False)

with open(OUT + "stop_conflict_stats.txt", "w") as f:
    f.write("CONFLICT INTENSITY: SUCCESS vs FAILED STOP\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Mean AUC:\n  SUCCESSFUL: {mean_succ:.4f}\n  FAILED: {mean_fail:.4f}\n  Diff: {mean_diff:+.4f}\n\n")
    f.write(f"t-test: t={t_stat:.3f}, p={p_val:.4f}, d={cohens_d:.3f}\n")

result = "YES" if p_val < 0.05 and abs(cohens_d) > 0.2 else "NO"
print(f"\nDimensionality tracks conflict: {result}")