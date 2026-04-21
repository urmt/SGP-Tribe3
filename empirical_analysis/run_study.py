#!/usr/bin/env python3
"""Load Gradient Study - Fixed"""
import os, gc, numpy as np, pandas as pd, nibabel as nib
from scipy import stats
from sklearn.decomposition import PCA

DATA_ROOT = "/home/student/sgp-tribe3/empirical_analysis/data/ds000030_s3/"
OUT = "/home/student/sgp-tribe3/empirical_analysis/multitask_results/"
TASKS = {'rest':1,'scap':2,'bart':3,'stopsignal':4,'taskswitch':5}

def dim(data, k):
    try:
        p = PCA(min(k+1, data.shape[0]-1)); p.fit(data)
        return np.sum(p.explained_variance_ratio_[:k])
    except: return np.nan

def proc(p):
    if not os.path.exists(p): return None
    img = nib.load(p)
    arr = img.get_fdata()
    n_tp = arr.shape[3]
    ts = arr.reshape(-1, n_tp).T.astype(np.float32)
    del arr, img; gc.collect()
    
    if ts.shape[0] < 20: return None
    ts = ts[:, ::max(1, ts.shape[1]//1500)]
    
    for j in range(ts.shape[1]):
        y = ts[:,j]; y = y - np.polyval(np.polyfit(np.arange(len(y)),y,1), np.arange(len(y)))
        ts[:,j] = (y-np.mean(y))/(np.std(y)+1e-10)
    
    d2 = dim(ts,2); d4 = dim(ts,4)
    del ts; gc.collect()
    return {'D2':d2,'D4':d4,'A':np.nanmean([d2,d4])}

print("Running 44 subjects, 5 tasks each...")
subjs = pd.read_csv("/home/student/sgp-tribe3/empirical_analysis/ds000030_qc/clean_subjects.csv")['subject'].tolist()
results = []
for si, s in enumerate(subjs):
    print(f"[{si+1}/44] {s}", flush=True)
    for t in TASKS:
        path = f"{DATA_ROOT}/{s}/func/{s}_task-{t}_bold.nii.gz"
        r = proc(path)
        if r:
            results.append({'subject':s,'task':t,'load':TASKS[t],'D2':r['D2'],'D4':r['D4'],'AUC':r['A']})
            print(".", end="", flush=True)
        else:
            print("X", end="", flush=True)
    gc.collect()
    if (si+1) % 10 == 0:
        print(f" [{si+1}/44 done]", flush=True)

df = pd.DataFrame(results)
df.to_csv(OUT+"auc_by_subject.csv", index=False)
print(f"\nSaved {len(df)} results", flush=True)

m = stats.linregress(df['load'], df['AUC'])
print(f"\nRegression: slope={m.slope:.5f} r={m.rvalue:.4f} p={m.pvalue:.5f}")

with open(OUT+"load_gradient_stats.txt","w") as f:
    f.write("LOAD GRADIENT ANALYSIS\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Regression: AUC ~ Load\n")
    f.write(f"  Slope: {m.slope:.5f}\n")
    f.write(f"  r: {m.rvalue:.4f}\n")
    f.write(f"  r²: {m.rvalue**2:.4f}\n")
    f.write(f"  p-value: {m.pvalue:.6f}\n\n")
    
    rest = df[df.task=='rest'].AUC
    f.write(f"REST baseline: mean={rest.mean():.4f}, sd={rest.std():.4f}\n\n")
    
    for t in ['scap','bart','stopsignal','taskswitch']:
        x = df[df.task==t].AUC
        tst, pv = stats.ttest_rel(x, rest)
        d = (x.mean()-rest.mean())/np.sqrt((x.var()+rest.var())/2)
        f.write(f"{t}: t={tst:.3f}, p={pv:.4f}, d={d:.3f}\n")
        print(f"  {t}: t={tst:.3f}, p={pv:.4f}, d={d:.3f}")

print(f"\n{'SUCCESS' if m.pvalue < 0.05 and m.slope > 0 else 'INCONCLUSIVE'}")