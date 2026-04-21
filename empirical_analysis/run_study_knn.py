#!/usr/bin/env python3
"""
Load Gradient Study - k-NN estimator version
"""
import os, gc, numpy as np, pandas as pd, nibabel as nib
from scipy import stats
from sklearn.neighbors import NearestNeighbors

DATA_ROOT = "/home/student/sgp-tribe3/empirical_analysis/data/ds000030_s3/"
OUT = "/home/student/sgp-tribe3/empirical_analysis/multitask_results/"
TASKS = {'rest':1,'scap':2,'bart':3,'stopsignal':4,'taskswitch':5}

def dim_knn(data, k):
    """k-NN dimensionality estimator"""
    try:
        n = data.shape[0]
        if n <= k + 1:
            return np.nan
        
        nn = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree')
        nn.fit(data)
        dists, _ = nn.kneighbors(data)
        
        log_dist = np.log(dists[:, -1] + 1e-10)
        return np.mean(log_dist)
    except:
        return np.nan

def proc(path):
    if not os.path.exists(path):
        return None
    
    img = nib.load(path)
    arr = img.get_fdata()
    n_tp = arr.shape[3]
    ts = arr.reshape(-1, n_tp).T.astype(np.float32)
    del arr, img
    gc.collect()
    
    if ts.shape[0] < 20:
        return None
    
    ts = ts[:, ::max(1, ts.shape[1]//1500)]
    
    for j in range(ts.shape[1]):
        y = ts[:, j]
        y = y - np.polyval(np.polyfit(np.arange(len(y)), y, 1), np.arange(len(y)))
        ts[:, j] = (y - np.mean(y)) / (np.std(y) + 1e-10)
    
    d2 = dim_knn(ts, 2)
    d4 = dim_knn(ts, 4)
    del ts
    gc.collect()
    
    return {'D2': d2, 'D4': d4, 'AUC': np.nanmean([d2, d4])}

print("=" * 60)
print("SFH-SGP_LOAD_GRADIENT_STUDY_02_REAL (k-NN)")
print("=" * 60)

subjs = pd.read_csv("/home/student/sgp-tribe3/empirical_analysis/ds000030_qc/clean_subjects.csv")['subject'].tolist()
results = []

for si, s in enumerate(subjs):
    print(f"[{si+1}/44] {s}", end=" ", flush=True)
    for t in TASKS:
        path = f"{DATA_ROOT}/{s}/func/{s}_task-{t}_bold.nii.gz"
        r = proc(path)
        if r:
            results.append({'subject': s, 'task': t, 'load': TASKS[t], 'D2': r['D2'], 'D4': r['D4'], 'AUC': r['AUC']})
            print(".", end="", flush=True)
        else:
            print("X", end="", flush=True)
    print(flush=True)
    gc.collect()

df = pd.DataFrame(results)
df.to_csv(OUT + "auc_by_subject.csv", index=False)
print(f"\nSaved {len(results)} results")

# Regression
m = stats.linregress(df['load'], df['AUC'])
print(f"\nRegression: slope={m.slope:.5f} r={m.rvalue:.4f} p={m.pvalue:.5f}")

# Pairwise
rest = df[df.task == 'rest'].AUC
with open(OUT + "load_gradient_stats.txt", "w") as f:
    f.write("LOAD GRADIENT ANALYSIS (k-NN)\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Regression: AUC ~ Load\n")
    f.write(f"  Slope: {m.slope:.5f}\n")
    f.write(f"  r: {m.rvalue:.4f}\n")
    f.write(f"  p: {m.pvalue:.6f}\n\n")
    f.write(f"REST: mean={rest.mean():.4f}, sd={rest.std():.4f}\n\n")
    
    for t in ['scap', 'bart', 'stopsignal', 'taskswitch']:
        x = df[df.task == t].AUC
       .tt, pval = stats.ttest_rel(x, rest)
        d = (x.mean() - rest.mean()) / np.sqrt((x.var() + rest.var()) / 2)
        f.write(f"{t}: t={tt:.3f}, p={pval:.4f}, d={d:.3f}\n")
        print(f"  {t}: t={tt:.3f}, p={pval:.4f}, d={d:.3f}")

# Bootstrap
print("\nBootstrap (1000)...")
boot_slopes = []
for _ in range(1000):
    sample = df.sample(n=len(df), replace=True)
    s, _, r, p, _ = stats.linregress(sample['load'], sample['AUC'])
    boot_slopes.append(s)
slope_ci = np.percentile(boot_slopes, [2.5, 97.5])
print(f"Slope 95% CI: [{slope_ci[0]:.5f}, {slope_ci[1]:.5f}]")

# LOSO
print("LOSO cross-validation...")
loso = []
for test_subj in subjs[:10]:
    train = df[df.subject != test_subj]
    s, _, r, p, _ = stats.linregress(train['load'], train['AUC'])
    loso.append(s)
print(f"Mean slope: {np.mean(loso):.5f} ± {np.std(loso):.5f}")

print(f"\n{'SUCCESS' if m.pvalue < 0.05 and m.slope > 0 else 'INCONCLUSIVE'}")