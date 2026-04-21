#!/usr/bin/env python3
"""
Control Confound Check: GO vs STOP signal properties
"""
import os, gc, numpy as np, pandas as pd, nibabel as nib
from scipy import stats

DATA_ROOT = "/home/student/sgp-tribe3/empirical_analysis/data/ds000030_s3/"
SUBJ_CSV = "/home/student/sgp-tribe3/empirical_analysis/ds000030_qc/clean_subjects.csv"
OUT = "/home/student/sgp-tribe3/empirical_analysis/multitask_results/"

print("=" * 60)
print("CONTROL CONFOUND CHECK: GO vs STOP")
print("=" * 60)

subjs = pd.read_csv(SUBJ_CSV)['subject'].tolist()
results = []

for si, s in enumerate(subjs):
    bold_path = f"{DATA_ROOT}/{s}/func/{s}_task-stopsignal_bold.nii.gz"
    events_path = f"{DATA_ROOT}/{s}/func/{s}_task-stopsignal_events.tsv"
    
    if not os.path.exists(bold_path) or not os.path.exists(events_path):
        continue
    
    try:
        events = pd.read_csv(events_path, sep='\t')
        
        go_mask = events['trial_type'] == 'GO'
        stop_mask = events['trial_type'] == 'STOP'
        
        go_onsets = events.loc[go_mask, 'onset'].values
        stop_onsets = events.loc[stop_mask, 'onset'].values
        
        if len(go_onsets) < 3 or len(stop_onsets) < 2:
            continue
        
        img = nib.load(bold_path)
        arr = img.get_fdata()
        n_tp = arr.shape[3]
        del img
        
        full_ts = arr.reshape(-1, n_tp).T.astype(np.float32)
        del arr
        gc.collect()
        
        # GO segments
        go_segs = []
        for onset in go_onsets[:15]:
            vol = int(onset * 2)
            if vol + 30 <= n_tp:
                go_segs.append(full_ts[vol:vol+30])
        
        # STOP segments  
        stop_segs = []
        for onset in stop_onsets[:10]:
            vol = int(onset * 2)
            if vol + 30 <= n_tp:
                stop_segs.append(full_ts[vol:vol+30])
        
        del full_ts
        gc.collect()
        
        if len(go_segs) < 2 or len(stop_segs) < 1:
            continue
        
        go_ts = np.vstack(go_segs)
        stop_ts = np.vstack(stop_segs)
        
        # Raw signal for confound check (before normalization)
        results.append({
            'subject': s,
            'n_tp_go': go_ts.shape[0],
            'n_tp_stop': stop_ts.shape[0],
            'mean_go': np.mean(go_ts),
            'mean_stop': np.mean(stop_ts),
            'var_go': np.var(go_ts),
            'var_stop': np.var(stop_ts),
            'std_go': np.std(go_ts),
            'std_stop': np.std(stop_ts),
            'range_go': np.max(go_ts) - np.min(go_ts),
            'range_stop': np.max(stop_ts) - np.min(stop_ts),
        })
        
        print(f"[{si+1}] {s}: GO n={go_ts.shape[0]}, STOP n={stop_ts.shape[0]}")
        print(f"       GO: mean={np.mean(go_ts):.1f}, var={np.var(go_ts):.1f}")
        print(f"       STOP: mean={np.mean(stop_ts):.1f}, var={np.var(stop_ts):.1f}")
        
        del go_ts, stop_ts
        gc.collect()
        
    except Exception as e:
        print(f"[{si+1}] {s}: ERROR {e}")

print(f"\nValid subjects: {len(results)}")

df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("CONFOUND ANALYSIS")
print("=" * 60)

# Number of timepoints
t_ntp, p_ntp = stats.ttest_rel(df['n_tp_go'], df['n_tp_stop'])
d_ntp = np.mean(df['n_tp_go'] - df['n_tp_stop']) / np.sqrt((df['n_tp_go'].var() + df['n_tp_stop'].var()) / 2)

print(f"\n1. Number of timepoints:")
print(f"   GO mean: {df['n_tp_go'].mean():.1f}")
print(f"   STOP mean: {df['n_tp_stop'].mean():.1f}")
print(f"   t={t_ntp:.3f}, p={p_ntp:.4f}, d={d_ntp:.3f}")

# Mean signal
t_mean, p_mean = stats.ttest_rel(df['mean_go'], df['mean_stop'])
d_mean = np.mean(df['mean_go'] - df['mean_stop']) / np.sqrt((df['mean_go'].var() + df['mean_stop'].var()) / 2)

print(f"\n2. Mean signal:")
print(f"   GO mean: {df['mean_go'].mean():.1f}")
print(f"   STOP mean: {df['mean_stop'].mean():.1f}")
print(f"   t={t_mean:.3f}, p={p_mean:.4f}, d={d_mean:.3f}")

# Variance
t_var, p_var = stats.ttest_rel(df['var_go'], df['var_stop'])
d_var = np.mean(df['var_go'] - df['var_stop']) / np.sqrt((df['var_go'].var() + df['var_stop'].var()) / 2)

print(f"\n3. Variance:")
print(f"   GO mean: {df['var_go'].mean():.1f}")
print(f"   STOP mean: {df['var_stop'].mean():.1f}")
print(f"   t={t_var:.3f}, p={p_var:.4f}, d={d_var:.3f}")

# Standard deviation
t_std, p_std = stats.ttest_rel(df['std_go'], df['std_stop'])
d_std = np.mean(df['std_go'] - df['std_stop']) / np.sqrt((df['std_go'].var() + df['std_stop'].var()) / 2)

print(f"\n4. Standard deviation:")
print(f"   GO mean: {df['std_go'].mean():.1f}")
print(f"   STOP mean: {df['std_stop'].mean():.1f}")
print(f"   t={t_std:.3f}, p={p_std:.4f}, d={d_std:.3f}")

# Range
t_range, p_range = stats.ttest_rel(df['range_go'], df['range_stop'])
d_range = np.mean(df['range_go'] - df['range_stop']) / np.sqrt((df['range_go'].var() + df['range_stop'].var()) / 2)

print(f"\n5. Signal range:")
print(f"   GO mean: {df['range_go'].mean():.1f}")
print(f"   STOP mean: {df['range_stop'].mean():.1f}")
print(f"   t={t_range:.3f}, p={p_range:.4f}, d={d_range:.3f}")

# Correlation with AUC
auc_df = pd.read_csv(OUT + "go_stop_contrast.csv")
merged = df.merge(auc_df, on='subject')

print("\n" + "=" * 60)
print("CORRELATION: Signal properties vs AUC difference")
print("=" * 60)

r_var, p_r = stats.pearsonr(merged['var_go'] - merged['var_stop'], merged['diff'])
print(f"\nVariance diff vs AUC diff: r={r_var:.3f}, p={p_r:.4f}")

r_std, p_rs = stats.pearsonr(merged['std_go'] - merged['std_stop'], merged['diff'])
print(f"STD diff vs AUC diff: r={r_std:.3f}, p={p_rs:.4f}")

r_ntp, p_ntpr = stats.pearsonr(merged['n_tp_go'] - merged['n_tp_stop'], merged['diff'])
print(f"Timepoints diff vs AUC diff: r={r_ntp:.3f}, p={p_ntpr:.4f}")

# Summary
print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)

confounds_found = []
if abs(d_ntp) > 0.2:
    confound = f"Timepoints: d={d_ntp:.3f}"
    if p_ntp < 0.05:
        confound += " (sig)"
    confounds_found.append(confound)

if abs(d_var) > 0.2:
    confound = f"Variance: d={d_var:.3f}"
    if p_var < 0.05:
        confound += " (sig)"
    confounds_found.append(confound)

if abs(d_std) > 0.2:
    confound = f"STD: d={d_std:.3f}"
    if p_std < 0.05:
        confound += " (sig)"
    confounds_found.append(confound)

if confounds_found:
    print("Systematic differences FOUND:")
    for c in confounds_found:
        print(f"  - {c}")
    independent = "NO"
else:
    print("No systematic confounds")
    independent = "YES"

print(f"\nDimensionality effect independent of signal differences: {independent}")