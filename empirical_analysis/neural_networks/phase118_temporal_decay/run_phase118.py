import os
import json
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import mne

BASE = "/home/student/sgp-tribe3"
OUTDIR = os.path.join(BASE, "empirical_analysis/neural_networks/phase118_temporal_decay")

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

SFREQ, WIN_SEC, WIN_SAMPLES = 128, 10, 1280
LAGS = [1, 2, 5]

def extract_features(x):
    x = np.nan_to_num(x, nan=0)
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
    return {'mean': np.mean(x), 'variance': np.var(x), 'hjorth_activity': act,
        'hjorth_mobility': mob, 'hjorth_complexity': comp, 'spectral_entropy': spec_ent,
        'bandpower_delta': band_pow['delta'], 'bandpower_theta': band_pow['theta'],
        'bandpower_alpha': band_pow['alpha'], 'bandpower_beta': band_pow['beta'], 'lag1_autocorr': lag1}

print("="*60)
print("PHASE 118 - TEMPORAL INFORMATION DECAY TEST")
print("="*60)

raw_signals = {}
for fpath in DATA_DIRS:
    if not os.path.exists(fpath): continue
    raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
    raw.resample(SFREQ)
    raw_signals[SUBJECT_MAP[os.path.basename(fpath)]] = raw.get_data()[0]

subjects = sorted(raw_signals.keys())
print(f"Subjects: {subjects}")

results = []
for lag in LAGS:
    print(f"\nLag: {lag}")
    real_r2s, null_r2s = [], []
    
    for ho_idx, ho in enumerate(subjects):
        train_subs = [s for s in subjects if s != ho]
        X_train, y_train = [], []
        
        for sub in train_subs:
            sig = raw_signals[sub]
            for i in range(0, len(sig) - (lag+1)*WIN_SAMPLES, WIN_SAMPLES):
                fc, fn = extract_features(sig[i:i+WIN_SAMPLES]), extract_features(sig[i+lag*WIN_SAMPLES:i+(lag+1)*WIN_SAMPLES])
                if fc and fn: X_train.append(list(fc.values())); y_train.append(list(fn.values()))
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        sig_test = raw_signals[ho]
        X_test, y_test = [], []
        for i in range(0, sig_test.shape[0] - (lag+1)*WIN_SAMPLES, WIN_SAMPLES):
            fc, fn = extract_features(sig_test[i:i+WIN_SAMPLES]), extract_features(sig_test[i+lag*WIN_SAMPLES:i+(lag+1)*WIN_SAMPLES])
            if fc and fn: X_test.append(list(fc.values())); y_test.append(list(fn.values()))
        
        X_test, y_test = np.array(X_test), np.array(y_test)
        
        clf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        clf.fit(X_train, y_train)
        real_r2s.append(r2_score(y_test, clf.predict(X_test)))
        
        rng = np.random.default_rng(42 + ho_idx)
        clf_perm = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        clf_perm.fit(X_train, rng.permutation(y_train))
        null_r2s.append(r2_score(y_test, clf_perm.predict(X_test)))
    
    real_mean, null_mean = np.mean(real_r2s), np.mean(null_r2s)
    results.append({'lag': lag, 'real_r2': real_mean, 'null_r2': null_mean, 'gain': real_mean - null_mean})
    print(f"  R²: {real_mean:.3f}, Null: {null_mean:.3f}")

df = pd.DataFrame(results)
df.to_csv(os.path.join(OUTDIR, "decay_results.csv"), index=False)

best = df.loc[df['real_r2'].idxmax()] if len(df) > 0 else df.iloc[0]
print("\n" + "="*60)
print(f"best_window: 10")
print(f"best_lag: {int(best['lag'])}")
print(f"best_r2: {best['real_r2']:.3f}")
print(f"null_r2: {best['null_r2']:.3f}")
print(f"predictive_gain: {best['gain']:.3f}")

verdict = "LOCAL_TEMPORAL_PERSISTENCE" if best['real_r2'] > 0 and best['lag'] <= 2 else "RAPID_DECAY_DYNAMICS"
print(f"VERDICT: {verdict}")

with open(os.path.join(OUTDIR, "final_summary.json"), "w") as f:
    json.dump({"best_window": 10, "best_lag": int(best['lag']), "best_r2": float(best['real_r2']),
        "null_r2": float(best['null_r2']), "predictive_gain": float(best['gain']), "verdict": verdict}, f, indent=2)

print("\nPHASE 118 RESULTS")
print(json.dumps({"best_window": 10, "best_lag": int(best['lag']), "best_r2": float(best['real_r2']),
    "null_r2": float(best['null_r2']), "predictive_gain": float(best['gain']), "verdict": verdict}, indent=2))