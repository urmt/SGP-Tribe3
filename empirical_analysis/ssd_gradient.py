#!/usr/bin/env python3
"""SSD Gradient Study - Full 44 subjects"""
import os, gc, numpy as np, pandas as pd, nibabel as nib
from scipy import stats
from sklearn.decomposition import PCA

DATA_ROOT = "/home/student/sgp-tribe3/empirical_analysis/data/ds000030_s3/"
SUBJ_CSV = "/home/student/sgp-tribe3/empirical_analysis/ds000030_qc/clean_subjects.csv"
OUT = "/home/student/sgp-tribe3/empirical_analysis/multitask_results/"

print("=" * 70)
print("SFH-SGP_SSD_GRADIENT_01 - FULL")
print("=" * 70)

def compute_dim(data, k):
    try:
        if data.shape[0] < k + 2 or data.shape[1] < 10:
            return np.nan
        p = PCA(min(k + 1, data.shape[0] - 1))
        p.fit(data)
        return np.sum(p.explained_variance_ratio_[:k])
    except:
        return np.nan

subjs = pd.read_csv(SUBJ_CSV)['subject'].tolist()
results = []

for si, s in enumerate(subjs):
    bold = f"{DATA_ROOT}/{s}/func/{s}_task-stopsignal_bold.nii.gz"
    events = f"{DATA_ROOT}/{s}/func/{s}_task-stopsignal_events.tsv"
    
    if not os.path.exists(bold) or not os.path.exists(events):
        continue
    
    ev = pd.read_csv(events, sep='\t')
    stop_mask = ev['TrialOutcome'].isin(['SuccessfulStop', 'UnsuccessfulStop'])
    stop_ev = ev[stop_mask]
    
    ssd_col = 'StopSignalDelay'
    valid = stop_ev[ssd_col] > 0
    ssds = stop_ev.loc[valid, ssd_col].values
    ons = stop_ev.loc[valid, 'onset'].values
    
    if len(ssds) < 12:
        continue
    
    median = np.median(ssds)
    low = ssds <= median
    high = ssds > median
    
    low_ons = ons[low][:20]
    high_ons = ons[high][:20]
    
    img = nib.load(bold)
    arr = img.get_fdata()
    n_tp = arr.shape[3]
    del img
    
    ts = arr.reshape(-1, n_tp).T.astype(np.float32)
    del arr
    gc.collect()
    
    def get_seg(onsets, ts, n_tp, mx=20):
        segs = []
        for i, o in enumerate(onsets):
            v = int(o * 2)
            if v + 15 <= n_tp and i * 20 < n_tp:
                segs.append(ts[v:v+15])
        return segs
    
    low_segs = get_seg(low_ons, ts, n_tp)
    high_segs = get_seg(high_ons, ts, n_tp)
    
    del ts
    gc.collect()
    
    if len(low_segs) < 2 or len(high_segs) < 2:
        continue
    
    low_ts = np.vstack(low_segs)
    high_ts = np.vstack(high_segs)
    
    for t in [low_ts, high_ts]:
        for j in range(min(500, t.shape[1])):
            y = t[:, j]
            t[:, j] = (y - np.mean(y)) / (np.std(y) + 1e-10)
    
    auc_low = np.nanmean([compute_dim(low_ts, 2), compute_dim(low_ts, 4)])
    auc_high = np.nanmean([compute_dim(high_ts, 2), compute_dim(high_ts, 4)])
    
    if not (np.isnan(auc_low) or np.isnan(auc_high)):
        results.append({
            'subject': s, 'auc_low': auc_low, 'auc_high': auc_high,
            'ssd_low': np.mean(ssds[low]), 'ssd_high': np.mean(ssds[high])
        })
        print(f"[{si+1}] {s}: LOW={auc_low:.3f} HIGH={auc_high:.3f}")
    
    del low_ts, high_ts
    gc.collect()

print(f"\nValid: {len(results)}")

df = pd.DataFrame(results)

m_low = df['auc_low'].mean()
m_high = df['auc_high'].mean()
print(f"\n{'Cond':<10} {'Mean AUC':>12} {'Std':>10}")
print(f"{'-'*34}")
print(f"{'LOW SSD':<10} {m_low:>12.4f} {df['auc_low'].std():>10.4f}")
print(f"{'HIGH SSD':<10} {m_high:>12.4f} {df['auc_high'].std():>10.4f}")

t, p = stats.ttest_rel(df['auc_low'], df['auc_high'])
d = (m_high - m_low) / df[['auc_low', 'auc_high']].stack().std()

print(f"\nPaired t-test: t={t:.3f}, p={p:.4f}, Cohen's d={d:.3f}")

# Save
df.to_csv(OUT + "ssd_gradient_results.csv", index=False)

with open(OUT + "ssd_gradient_stats.txt", "w") as f:
    f.write("SSD GRADIENT STUDY\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Valid subjects: {len(df)}\n\n")
    f.write(f"Mean AUC:\n  LOW SSD: {m_low:.4f}\n  HIGH SSD: {m_high:.4f}\n\n")
    f.write(f"Paired t-test: t={t:.3f}, p={p:.4f}, d={d:.3f}\n")

result = "YES" if (p < 0.05 and d > 0) else "NO"
print(f"\nDimensionality increases with SSD: {result}")