#!/usr/bin/env python3
"""
Load Gradient Study - BART Within-Task Variation
Test dimensionality vs temporal progression within BART task
"""
import os, gc, numpy as np, pandas as pd, nibabel as nib
from scipy import stats

DATA_ROOT = "/home/student/sgp-tribe3/empirical_analysis/data/ds000030_s3/"
SUBJ_CSV = "/home/student/sgp-tribe3/empirical_analysis/ds000030_qc/clean_subjects.csv"
OUT = "/home/student/sgp-tribe3/empirical_analysis/multitask_results/"

def compute_dimensionality(data, k):
    """PCA-based dimensionality (sum of first k eigenvalues)"""
    try:
        if data.shape[0] < k + 2:
            return np.nan
        from sklearn.decomposition import PCA
        p = PCA(min(k + 1, data.shape[0] - 1))
        p.fit(data)
        return np.sum(p.explained_variance_ratio_[:k])
    except:
        return np.nan

def process_segment(arr, seg_idx, n_segs=3):
    """Extract and process one temporal segment"""
    n_tp = arr.shape[3]
    seg_size = n_tp // n_segs
    
    if seg_size < 30:
        return None
    
    start = seg_idx * seg_size
    end = start + seg_size
    
    ts = arr.reshape(-1, n_tp).T[start:end].astype(np.float32)
    
    ts = ts[:, ::max(1, ts.shape[1] // 1500)]
    
    for j in range(ts.shape[1]):
        y = ts[:, j]
        y = y - np.polyval(np.polyfit(np.arange(len(y)), y, 1), np.arange(len(y)))
        ts[:, j] = (y - np.mean(y)) / (np.std(y) + 1e-10)
    
    return ts

print("=" * 60)
print("BART WITHIN-TASK LOAD GRADIENT ANALYSIS")
print("=" * 60)

subjs = pd.read_csv(SUBJ_CSV)['subject'].tolist()
print(f"Subjects: {len(subjs)}")

results = []
failures = []

for si, s in enumerate(subjs):
    path = f"{DATA_ROOT}/{s}/func/{s}_task-bart_bold.nii.gz"
    
    if not os.path.exists(path):
        print(f"[{si+1}] {s}: FILE NOT FOUND")
        failures.append((s, "not found"))
        continue
    
    try:
        img = nib.load(path)
        arr = img.get_fdata()
        del img
        gc.collect()
        
        n_tp = arr.shape[3]
        
        if n_tp < 100:
            print(f"[{si+1}] {s}: TOO SHORT ({n_tp}vol)")
            failures.append((s, "too short"))
            continue
        
        segments = []
        for seg_idx in range(3):
            ts = process_segment(arr, seg_idx)
            if ts is None or ts.shape[0] < 30:
                segments.append(None)
                continue
            
            d2 = compute_dimensionality(ts, 2)
            d4 = compute_dimensionality(ts, 4)
            d8 = compute_dimensionality(ts, 8)
            d16 = compute_dimensionality(ts, 16)
            auc = np.nanmean([d2, d4])
            
            segments.append({
                'D2': d2, 'D4': d4, 'D8': d8, 'D16': d16, 'AUC': auc
            })
            
            del ts
            gc.collect()
        
        del arr
        gc.collect()
        
        if all(s is not None for s in segments):
            results.append({
                'subject': s,
                'auc_low': segments[0]['AUC'],
                'auc_mid': segments[1]['AUC'],
                'auc_high': segments[2]['AUC']
            })
            print(f"[{si+1}] {s}: LOW={segments[0]['AUC']:.3f} MID={segments[1]['AUC']:.3f} HIGH={segments[2]['AUC']:.3f}")
        else:
            failures.append((s, "segment error"))
            
    except Exception as e:
        print(f"[{si+1}] {s}: ERROR {e}")
        failures.append((s, str(e)))

print(f"\nValid subjects: {len(results)}, Failures: {len(failures)}")

df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("STEP 5: WITHIN-SUBJECT ANALYSIS")
print("=" * 60)

mean_low = df['auc_low'].mean()
mean_mid = df['auc_mid'].mean()
mean_high = df['auc_high'].mean()

print(f"\nMean AUC:")
print(f"  LOW (early):  {mean_low:.4f}")
print(f"  MID:         {mean_mid:.4f}")
print(f"  HIGH (late): {mean_high:.4f}")

print("\n" + "=" * 60)
print("STEP 6: GROUP ANALYSIS - LINEAR TREND")
print("=" * 60)

load_levels = np.array([1, 2, 3])
subject_means = []
for _, row in df.iterrows():
    subject_means.append([row['auc_low'], row['auc_mid'], row['auc_high']])

subject_means = np.array(subject_means)
subject_flat = []
for subj_vals in subject_means:
    subject_flat.extend(subj_vals)

load_flat = np.tile(load_levels, len(df))

slope, intercept, r, p, se = stats.linregress(load_flat, subject_flat)

print(f"\nLinear trend: AUC ~ Load")
print(f"  Slope: {slope:.5f}")
print(f"  r: {r:.4f}")
print(f"  p-value: {p:.4f}")

print("\n" + "=" * 60)
print("PAIRWISE COMPARISONS (paired t-tests)")
print("=" * 60)

t_low_mid, p_lm = stats.ttest_rel(df['auc_low'], df['auc_mid'])
t_mid_high, p_mh = stats.ttest_rel(df['auc_mid'], df['auc_high'])
t_low_high, p_lh = stats.ttest_rel(df['auc_low'], df['auc_high'])

d_low_mid = (df['auc_mid'].mean() - df['auc_low'].mean()) / df[['auc_low', 'auc_mid']].stack().std()
d_mid_high = (df['auc_high'].mean() - df['auc_mid'].mean()) / df[['auc_mid', 'auc_high']].stack().std()
d_low_high = (df['auc_high'].mean() - df['auc_low'].mean()) / df[['auc_low', 'auc_high']].stack().std()

print(f"\nLOW vs MID:  t={t_low_mid:.3f}, p={p_lm:.4f}, d={d_low_mid:.3f}")
print(f"MID vs HIGH: t={t_mid_high:.3f}, p={p_mh:.4f}, d={d_mid_high:.3f}")
print(f"LOW vs HIGH: t={t_low_high:.3f}, p={p_lh:.4f}, d={d_low_high:.3f}")

df.to_csv(OUT + "bart_within_task_results.csv", index=False)

with open(OUT + "load_gradient_stats.txt", "w") as f:
    f.write("BART WITHIN-TASK LOAD GRADIENT\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Mean AUC:\n")
    f.write(f"  LOW:  {mean_low:.4f}\n")
    f.write(f"  MID:  {mean_mid:.4f}\n")
    f.write(f"  HIGH: {mean_high:.4f}\n\n")
    f.write(f"Linear Trend:\n")
    f.write(f"  Slope: {slope:.5f}\n")
    f.write(f"  r: {r:.4f}\n")
    f.write(f"  p-value: {p:.4f}\n\n")
    f.write(f"Pairwise:\n")
    f.write(f"  LOW vs MID: t={t_low_mid:.3f}, p={p_lm:.4f}, d={d_low_mid:.3f}\n")
    f.write(f"  MID vs HIGH: t={t_mid_high:.3f}, p={p_mh:.4f}, d={d_mid_high:.3f}\n")
    f.write(f"  LOW vs HIGH: t={t_low_high:.3f}, p={p_lh:.4f}, d={d_low_high:.3f}\n")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

monotonic_increasing = (mean_low < mean_mid < mean_high) or (mean_low < mean_high and mean_mid < mean_high)
significant = p < 0.05 and slope > 0

if monotonic_increasing and significant:
    print("Dimensionality INCREASES with load: YES")
    print(f"Trend: {mean_low:.3f} < {mean_mid:.3f} < {mean_high:.3f}")
    print(f"Statistics: slope={slope:.4f}, r={r:.3f}, p={p:.4f}")
    result = "SUCCESS"
elif p < 0.05:
    print(f"Dimensionality changes with load: YES (p={p:.4f})")
    print(f"Direction: {'positive' if slope > 0 else 'negative'}")
    result = "PARTIAL"
else:
    print("Dimensionality INCREASES with load: NO")
    print(f"Trend: LOW={mean_low:.3f}, MID={mean_mid:.3f}, HIGH={mean_high:.3f}")
    result = "INCONCLUSIVE"

print(f"\nFinal: {result}")