import os
import json
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import mne

BASE = "/home/student/sgp-tribe3"
OUTDIR = os.path.join(BASE, "empirical_analysis/neural_networks/phase117_causal_predictability")

DATA_DIRS = [
    os.path.join(BASE, "empirical_analysis/neural_networks/phase105_real_eeg_download/raw/CHBMIT.edf"),
    os.path.join(BASE, "empirical_analysis/neural_networks/phase111_long_duration_real_eeg/downloaded/chb01_03.edf"),
    os.path.join(BASE, "empirical_analysis/neural_networks/phase112_persistent_acquisition/downloaded/chb01_04.edf"),
    os.path.join(BASE, "empirical_analysis/neural_networks/phase112_persistent_acquisition/downloaded/chb02_01.edf"),
    os.path.join(BASE, "empirical_analysis/neural_networks/phase112_persistent_acquisition/downloaded/chb03_01.edf"),
    os.path.join(BASE, "empirical_analysis/neural_networks/phase112_persistent_acquisition/downloaded/chb04_01.edf"),
]

SUBJECT_MAP = {"CHBMIT.edf": "chb00", "chb01_03.edf": "chb01", "chb01_04.edf": "chb01",
    "chb02_01.edf": "chb02", "chb03_01.edf": "chb03", "chb04_01.edf": "chb04"}

WINDOW_SEC, SFREQ = 10, 128

def iaaft_surrogate(x, rng, n_iter=25):
    x = x.astype(float)
    n = len(x)
    sorted_x = np.sort(x)
    spectrum = np.abs(np.fft.fft(x))
    x_new = rng.standard_normal(n)
    for _ in range(n_iter):
        phases = rng.uniform(-np.pi, np.pi, n)
        fourier = spectrum * np.exp(1j * phases)
        x_phase = np.real(np.fft.ifft(fourier))
        sorted_phase = np.sort(x_phase)
        x_new = sorted_x[np.searchsorted(sorted_phase, x_phase)]
    return x_new

def extract_features(x):
    x = np.nan_to_num(x, nan=0, posinf=0, neginf=0)
    if np.std(x) < 1e-10: return None
    f, p = signal.welch(x, fs=SFREQ, nperseg=min(256, len(x)))
    p = np.abs(p) + 1e-12
    bands = {"delta": (1,4), "theta": (4,8), "alpha": (8,12), "beta": (12,30)}
    band_pow = {n: np.mean(p[(f>=l)&(f<h)]) for n,(l,h) in bands.items()}
    p_norm = p / (np.sum(p) + 1e-12)
    spec_ent = -np.sum(p_norm * np.log(p_norm + 1e-12))
    dx = np.diff(x)
    act = np.var(x)
    mob = np.sqrt(np.var(dx) / (act + 1e-12))
    comp = np.sqrt(np.var(np.diff(dx)) / (np.var(dx) + 1e-12)) / (mob + 1e-12)
    try: lag1 = np.corrcoef(x[:-1], x[1:])[0,1]
    except: lag1 = 0
    zcr = np.sum(np.diff(np.sign(x)) != 0) / len(x)
    return {'mean': float(np.mean(x)), 'variance': float(np.var(x)), 'RMS': float(np.sqrt(np.mean(x**2))),
        'hjorth_activity': float(act), 'hjorth_mobility': float(mob), 'hjorth_complexity': float(comp),
        'spectral_entropy': float(spec_ent), 'bandpower_delta': float(band_pow['delta']),
        'bandpower_theta': float(band_pow['theta']), 'bandpower_alpha': float(band_pow['alpha']),
        'bandpower_beta': float(band_pow['beta']), 'lag1_autocorr': float(lag1), 'zero_crossing_rate': float(zcr)}

print("="*60)
print("PHASE 117 - CAUSAL PREDICTABILITY TEST")
print("="*60)

raw_signals = {}

for fpath in DATA_DIRS:
    if not os.path.exists(fpath): continue
    fname = os.path.basename(fpath)
    subj = SUBJECT_MAP.get(fname, fname.split('_')[0])
    raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
    raw.resample(SFREQ)
    raw_signals[subj] = raw.get_data()[0]

subjects = sorted(raw_signals.keys())
print(f"Subjects: {subjects}")

real_results, iaaft_results, perm_results = [], [], []
fold_results = []

for ho_idx, ho in enumerate(subjects):
    print(f"\nFold {ho_idx+1}: hold out {ho}")
    train_subs = [s for s in subjects if s != ho]
    
    X_train, y_train = [], []
    X_iaaft, y_iaaft = [], []
    
    for sub in train_subs:
        sig = raw_signals[sub]
        rng = np.random.default_rng(42 + hash(sub) % 1000)
        win_len = WINDOW_SEC * SFREQ
        
        feats_current = []
        feats_next = []
        feats_iaaft_current = []
        feats_iaaft_next = []
        
        for st in range(0, len(sig) - 2*win_len, win_len):
            seg_curr = sig[st:st+win_len]
            seg_next = sig[st+win_len:st+2*win_len]
            
            feat_curr = extract_features(seg_curr)
            feat_next = extract_features(seg_next)
            
            if feat_curr and feat_next:
                feats_current.append([feat_curr[k] for k in sorted(feat_curr.keys())])
                feats_next.append([feat_next[k] for k in sorted(feat_next.keys())])
                
                seg_iaaft_curr = iaaft_surrogate(seg_curr, rng, 25)
                seg_iaaft_next = iaaft_surrogate(seg_next, rng, 25)
                
                feat_iaaft_curr = extract_features(seg_iaaft_curr)
                feat_iaaft_next = extract_features(seg_iaaft_next)
                
                if feat_iaaft_curr and feat_iaaft_next:
                    feats_iaaft_current.append([feat_iaaft_curr[k] for k in sorted(feat_iaaft_curr.keys())])
                    feats_iaaft_next.append([feat_iaaft_next[k] for k in sorted(feat_iaaft_next.keys())])
        
        X_train.extend(feats_current)
        y_train.extend(feats_next)
        X_iaaft.extend(feats_iaaft_current)
        y_iaaft.extend(feats_iaaft_next)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_iaaft = np.array(X_iaaft)
    y_iaaft = np.array(y_iaaft)
    
    sig_test = raw_signals[ho]
    rng_test = np.random.default_rng(42 + ho_idx)
    win_len = WINDOW_SEC * SFREQ
    
    X_test, y_test = [], []
    X_iaaft_test, y_iaaft_test = [], []
    
    for st in range(0, len(sig_test) - 2*win_len, win_len):
        seg_curr = sig_test[st:st+win_len]
        seg_next = sig_test[st+win_len:st+2*win_len]
        
        feat_curr = extract_features(seg_curr)
        feat_next = extract_features(seg_next)
        
        if feat_curr and feat_next:
            X_test.append([feat_curr[k] for k in sorted(feat_curr.keys())])
            y_test.append([feat_next[k] for k in sorted(feat_next.keys())])
            
            seg_iaaft_curr = iaaft_surrogate(seg_curr, rng_test, 25)
            seg_iaaft_next = iaaft_surrogate(seg_next, rng_test, 25)
            
            feat_iaaft_curr = extract_features(seg_iaaft_curr)
            feat_iaaft_next = extract_features(seg_iaaft_next)
            
            if feat_iaaft_curr and feat_iaaft_next:
                X_iaaft_test.append([feat_iaaft_curr[k] for k in sorted(feat_iaaft_curr.keys())])
                y_iaaft_test.append([feat_iaaft_next[k] for k in sorted(feat_iaaft_next.keys())])
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_iaaft_test = np.array(X_iaaft_test)
    y_iaaft_test = np.array(y_iaaft_test)
    
    clf_real = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    clf_real.fit(X_train, y_train)
    pred_real = clf_real.predict(X_test)
    real_r2 = r2_score(y_test, pred_real)
    real_results.append(real_r2)
    print(f"  Real R²: {real_r2:.3f}")
    
    clf_iaaft = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    clf_iaaft.fit(np.vstack([X_train, X_iaaft]), np.vstack([y_train, y_iaaft]))
    pred_iaaft = clf_iaaft.predict(X_iaaft_test)
    iaaft_r2 = r2_score(y_iaaft_test, pred_iaaft)
    iaaft_results.append(iaaft_r2)
    print(f"  IAAFT R²: {iaaft_r2:.3f}")
    
    rng_perm = np.random.default_rng(100 + ho_idx)
    y_perm = rng_perm.permutation(y_train)
    clf_perm = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    clf_perm.fit(X_train, y_perm)
    pred_perm = clf_perm.predict(X_test)
    perm_r2 = r2_score(y_test, pred_perm)
    perm_results.append(perm_r2)
    print(f"  Perm R²: {perm_r2:.3f}")
    
    fold_results.append({'fold': ho_idx+1, 'held_out': ho, 'real_r2': real_r2, 'iaaft_r2': iaaft_r2, 'perm_r2': perm_r2})

real_mean = float(np.mean(real_results))
iaaft_mean = float(np.mean(iaaft_results))
perm_mean = float(np.mean(perm_results))
gain = real_mean - iaaft_mean

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"real_r2: {real_mean:.3f}")
print(f"iaaft_r2: {iaaft_mean:.3f}")
print(f"perm_r2: {perm_mean:.3f}")
print(f"predictive_gain: {gain:.3f}")

if real_mean > iaaft_mean + 0.1:
    interp = "Real EEG has genuine causal temporal dynamics - past predicts future better than IAAFT controls"
    verdict = "GENUINE_TEMPORAL_DYNAMICS"
elif real_mean > 0 and iaaft_mean > 0:
    interp = "Both real and IAAFT show predictability - organization reducible to stationary nonlinear statistics"
    verdict = "MOSTLY_STATIONARY_NONLINEAR"
else:
    interp = "No robust predictive structure found"
    verdict = "NO_ROBUST_PREDICTABILITY"

print(f"\nINTERPRETATION: {interp}")
print(f"VERDICT: {verdict}")

pd.DataFrame(fold_results).to_csv(os.path.join(OUTDIR, "fold_results.csv"), index=False)

result = {"real_r2": real_mean, "iaaft_r2": iaaft_mean, "perm_r2": perm_mean,
    "predictive_gain": gain, "interpretation": interp, "verdict": verdict, "n_subjects": len(subjects)}

with open(os.path.join(OUTDIR, "final_summary.json"), "w") as f:
    json.dump(result, f, indent=2)

print("\nPHASE 117 RESULTS")
print(json.dumps(result, indent=2))